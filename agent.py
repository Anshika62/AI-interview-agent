"""LiveKit Agent Worker conducting 5-stage personalized mock interviews."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

# Path setup to resolve prompts and backend modules cleanly
_here = Path(__file__).resolve().parent
_root = _here.parent
_backend = _root / "backend"
for p in [str(_here), str(_root), str(_backend)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Load environment configuration from agents, root, and backend
load_dotenv(_here / ".env", override=False)
load_dotenv(_root / ".env", override=False)
load_dotenv(_backend / ".env", override=False)

from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli, function_tool, voice

# CRITICAL: Pre-register plugins on the main thread during module load.
# LiveKit Agent worker threads will raise RuntimeError if plugins are imported lazily inside job tasks.
from livekit.plugins import openai

try:
    from livekit.plugins import silero
except Exception as _e:
    silero = None

try:
    from livekit.plugins import deepgram
except Exception as _e:
    deepgram = None

try:
    from livekit.plugins import cartesia
except Exception as _e:
    cartesia = None

try:
    from livekit.plugins import elevenlabs
except Exception as _e:
    elevenlabs = None

from prompts import build_interview_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_interview_agent.worker")

BACKEND_API_URL = (os.getenv("BACKEND_API_URL") or os.getenv("BASE_URL") or "http://localhost:8000").rstrip("/")
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME") or "AI-interview-agent-local"

# Thread/loop-safe HTTP client pool to avoid SSL context recreation and cross-event-loop lock errors
_http_clients: Dict[int, httpx.AsyncClient] = {}


def get_http_client() -> httpx.AsyncClient:
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = 0
    client = _http_clients.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=8.0, verify=False)
        _http_clients[loop_id] = client
    return client


def create_pipeline_components(language: str = "English") -> Dict[str, Any]:
    """
    Initialize STT, TTS, LLM, and VAD components with rock-solid fallbacks.
    Guarantees that LLM and TTS are never None.
    Supports multi-language STT & TTS automatically based on interview language.
    All keys, models, and endpoints are loaded exclusively from .env.
    """
    try:
        from backend.app.services.question_assistant import resolve_language_config
        lang_cfg = resolve_language_config(language)
    except Exception:
        lang_cfg = {"name": language or "English", "code": "en-US", "deepgram_code": "en-US"}

    stt_language = os.getenv("DEEPGRAM_LANGUAGE") or lang_cfg.get("deepgram_code", "en-US")
    if language and language.strip().lower() not in ("english", "en", "en-us"):
        stt_language = lang_cfg.get("deepgram_code", "multi")

    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key and (not openai_key or openai_key.startswith("your_") or openai_key.startswith("sk-placeholder")):
        logger.warning("Neither GROQ_API_KEY nor valid OPENAI_API_KEY is configured in .env.")

    # 1. Voice Activity Detection (Silero VAD)
    vad = None
    if silero is not None:
        try:
            vad = silero.VAD.load(
                min_silence_duration=float(os.getenv("VAD_MIN_SILENCE_DURATION") or "0.4"),
                activation_threshold=float(os.getenv("VAD_ACTIVATION_THRESHOLD") or "0.5"),
                deactivation_threshold=float(os.getenv("VAD_DEACTIVATION_THRESHOLD") or "0.35"),
            )
            logger.info("Silero VAD loaded successfully")
        except Exception as e:
            logger.warning(f"Silero VAD initialization failed ({e}), using default turn detector")

    # 2. Speech-to-Text (STT) - Deepgram preferred (supports multi-language), OpenAI STT fallback
    stt = None
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    deepgram_model = os.getenv("DEEPGRAM_MODEL") or "nova-3"
    if deepgram is not None and deepgram_key and not deepgram_key.startswith("your_"):
        try:
            stt = deepgram.STT(
                api_key=deepgram_key,
                model=deepgram_model,
                language=stt_language,
            )
            logger.info(f"Deepgram STT ({deepgram_model}, lang={stt_language}) initialized successfully")
        except Exception as e:
            logger.warning(f"Deepgram STT failed ({e}), checking OpenAI STT fallback")

    if stt is None and openai_key:
        try:
            stt = openai.STT(api_key=openai_key)
            logger.info("OpenAI STT initialized as fallback")
        except Exception as e:
            logger.warning(f"OpenAI STT initialization error: {e}")

    # 3. Text-to-Speech (TTS) - ElevenLabs -> Cartesia -> OpenAI TTS fallback
    tts = None
    cartesia_key = os.getenv("CARTESIA_API_KEY")
    if cartesia is not None and cartesia_key and not cartesia_key.startswith("your_"):
        cartesia_model = os.getenv("CARTESIA_MODEL") or "sonic-english"
        cartesia_voice = os.getenv("CARTESIA_VOICE") or "248be419-c632-4f23-adf1-5324ed7dbf10"
        try:
            tts = cartesia.TTS(
                api_key=cartesia_key,
                model=cartesia_model,
                voice=cartesia_voice,
            )
            logger.info("Cartesia TTS initialized successfully")
        except Exception as e:
            logger.warning(f"Cartesia TTS failed ({e}), checking ElevenLabs fallback")

    if tts is None:
        eleven_key = os.getenv("ELEVENLABS_API_KEY")
        if elevenlabs is not None and eleven_key and not eleven_key.startswith("your_"):
            eleven_voice_id = os.getenv("ELEVENLABS_VOICE_ID") or "pNInz6obpgDQGcFmaJgB"
            eleven_model = os.getenv("ELEVENLABS_MODEL") or "eleven_flash_v2_5"
            try:
                tts = elevenlabs.TTS(
                    api_key=eleven_key,
                    voice_id=eleven_voice_id,
                    model=eleven_model,
                )
                logger.info("ElevenLabs TTS initialized successfully")
            except Exception as e:
                logger.warning(f"ElevenLabs TTS failed ({e}), falling back to OpenAI TTS")

    if tts is None and openai_key:
        try:
            tts = openai.TTS(api_key=openai_key, voice="alloy")
            logger.info("OpenAI TTS initialized as reliable fallback")
        except Exception as e:
            logger.warning(f"OpenAI TTS fallback error: {e}")

    # 4. Large Language Model (LLM) - Groq (preferred) or OpenAI fallback
    llm = None
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and not groq_key.startswith("your_") and not groq_key.startswith("sk-placeholder") and len(groq_key.strip()) > 5:
        model = os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
        base_url = os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
        try:
            llm = openai.LLM(
                api_key=groq_key,
                base_url=base_url,
                model=model,
                temperature=0.3,
            )
            logger.info(f"Groq LLM ({model}) initialized successfully via {base_url}")
        except Exception as e:
            logger.warning(f"Groq LLM init error ({e}), checking OpenAI LLM fallback")

    if llm is None and openai_key and not openai_key.startswith("your_") and not openai_key.startswith("sk-placeholder"):
        model = os.getenv("OPENAI_MODEL") or "gpt-4o"
        try:
            llm = openai.LLM(
                api_key=openai_key,
                model=model,
                temperature=0.3,
            )
            logger.info(f"OpenAI LLM ({model}) initialized successfully")
        except Exception as e:
            logger.error(f"OpenAI LLM init error: {e}")

    if llm is None:
        raise RuntimeError("LLM component could not be initialized. Please set GROQ_API_KEY or OPENAI_API_KEY in .env.")

    if tts is None:
        raise RuntimeError("TTS component could not be initialized. Check ELEVENLABS_API_KEY or OPENAI_API_KEY.")

    return {
        "vad": vad,
        "stt": stt,
        "tts": tts,
        "llm": llm,
    }


async def fetch_candidate_profile(candidate_id: str) -> Optional[Dict[str, Any]]:
    """Fetch candidate profile from backend API by candidate_id, email, or prefix."""
    if not candidate_id:
        return None
    url = f"{BACKEND_API_URL}/candidate/{candidate_id}"
    try:
        client = get_http_client()
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            cand = data.get("candidate") or data
            logger.info(f"Retrieved candidate profile for '{candidate_id}': {cand.get('name')}")
            return cand
    except Exception as e:
        logger.debug(f"Could not fetch candidate profile for '{candidate_id}': {e}")
    return None


async def send_transcript_turn(candidate_id: str, room_name: str, speaker: str, text: str) -> None:
    """Record speech turn on the backend transcript store."""
    url = f"{BACKEND_API_URL}/transcript"
    try:
        client = get_http_client()
        await client.post(
            url,
            json={
                "candidate_id": candidate_id,
                "room_name": room_name,
                "speaker": speaker,
                "text": text,
            },
        )
    except Exception as e:
        logger.debug(f"Could not record transcript turn: {e}")


async def run_interview_agent(ctx: JobContext) -> None:
    """Run voice interview session inside LiveKit room."""
    room_name = ctx.room.name
    logger.info(f"Connected interview agent to room: '{room_name}'")

    # Extract candidate_id from room_name (e.g. 'mock-interview-8844073c-d862' -> '8844073c')
    raw_id = room_name.replace("mock-interview-", "").replace("room-", "")
    candidate_id = raw_id

    # 1. Inspect metadata from LiveKit Cloud console dispatch or room metadata
    job_metadata = getattr(ctx, "job", None) and getattr(ctx.job, "metadata", None)
    raw_meta = job_metadata or ctx.room.metadata

    candidate: Optional[Dict[str, Any]] = None
    if raw_meta:
        logger.info(f"Received dispatch/room metadata: {raw_meta}")
        try:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            if isinstance(meta, dict):
                if meta.get("skills") or meta.get("projects"):
                    candidate = meta
                if meta.get("candidate_id"):
                    candidate_id = meta["candidate_id"]
                elif meta.get("email"):
                    candidate_id = meta["email"]
        except Exception:
            if isinstance(raw_meta, str) and raw_meta.strip():
                candidate_id = raw_meta.strip()

    # 2. If candidate profile was not fully embedded in metadata, fetch from backend API
    if not candidate or not candidate.get("skills"):
        fetched = await fetch_candidate_profile(candidate_id)
        if not fetched and "-" in candidate_id:
            possible_id = candidate_id.rsplit("-", 1)[0]
            fetched = await fetch_candidate_profile(possible_id)
            if fetched:
                candidate_id = possible_id
        if fetched:
            candidate = fetched

    # 3. Check if candidate participant is already in the room with metadata
    for p in ctx.room.remote_participants.values():
        if not p.identity.startswith("agent-"):
            if not candidate or not candidate.get("skills"):
                cand_by_pid = await fetch_candidate_profile(p.identity)
                if cand_by_pid and cand_by_pid.get("skills"):
                    candidate = cand_by_pid
                    candidate_id = candidate.get("candidate_id", candidate_id)
                    logger.info(f"Loaded candidate info by participant identity: {candidate.get('name')}")

            if p.metadata and (not candidate or not candidate.get("skills")):
                try:
                    p_meta = json.loads(p.metadata)
                    if isinstance(p_meta, dict) and (p_meta.get("skills") or p_meta.get("projects")):
                        candidate = p_meta
                        logger.info(f"Loaded candidate info from remote participant metadata: {candidate.get('name')}")
                except Exception:
                    pass

    # 4. Fallback profile if no resume or skills were provided
    if not candidate:
        candidate = {
            "candidate_id": candidate_id or "default_candidate",
            "name": candidate_id.capitalize() if candidate_id and not candidate_id.startswith("cand_") else "Candidate",
            "skills": ["Python", "FastAPI", "PostgreSQL", "SQLAlchemy", "LiveKit"],
            "projects": [
                {
                    "title": "Backend Microservice Platform",
                    "description": "High-throughput asynchronous REST API built with FastAPI and PostgreSQL",
                    "technologies": ["Python", "FastAPI", "PostgreSQL"],
                },
                {
                    "title": "Real-time Voice Communication",
                    "description": "Interactive WebRTC audio interviewer platform using LiveKit",
                    "technologies": ["Python", "LiveKit", "WebSockets"],
                },
            ],
            "experience": "2+ years of backend development experience",
        }

    candidate_name = candidate.get("name", "Candidate")
    skills = candidate.get("skills", [])
    interview_role = candidate.get("role") or "Software Engineer"
    interview_level = candidate.get("position_level") or "Mid-level"
    interview_language = candidate.get("language") or "English"
    interview_jd = candidate.get("job_description") or ""
    logger.info(
        f"Configuring mock interview for {candidate_name} | Role: {interview_role} "
        f"({interview_level}) | Language: {interview_language} | Skills: {skills}"
    )

    # Event to keep the session active until candidate finishes or disconnects
    session_done = asyncio.Event()

    # Candidate Silence / Stuck Watchdog Settings (configurable via .env)
    candidate_response_timeout = float(os.getenv("CANDIDATE_RESPONSE_TIMEOUT", "8"))
    candidate_max_silence_timeout = float(os.getenv("CANDIDATE_MAX_SILENCE_TIMEOUT", "18"))
    if candidate_max_silence_timeout <= candidate_response_timeout:
        candidate_max_silence_timeout = candidate_response_timeout + 10.0

    silence_watchdog_task: Optional[asyncio.Task] = None
    is_watchdog_speaking: bool = False
    interview_active: bool = False
    is_english = interview_language.strip().lower() in ("english", "en", "en-us", "")

    def cancel_silence_watchdog(reason: str = "") -> None:
        nonlocal silence_watchdog_task
        if silence_watchdog_task and not silence_watchdog_task.done():
            logger.info("Cancelling silence watchdog (%s)", reason)
            silence_watchdog_task.cancel()
        silence_watchdog_task = None

    def start_silence_watchdog() -> None:
        nonlocal silence_watchdog_task
        if not interview_active or session_done.is_set() or is_watchdog_speaking:
            return
        if silence_watchdog_task and not silence_watchdog_task.done():
            return
        silence_watchdog_task = asyncio.create_task(_run_silence_watchdog())

    async def _run_silence_watchdog() -> None:
        nonlocal is_watchdog_speaking
        try:
            logger.info(
                "Silence watchdog active: waiting up to %.1fs for candidate response before nudge...",
                candidate_response_timeout,
            )
            await asyncio.sleep(candidate_response_timeout)

            # Stage 1: Check conditions before nudge
            if session_done.is_set():
                return
            if agent_session.user_state == "speaking":
                logger.info("Candidate started speaking; silence nudge aborted.")
                return

            # Stage 1: Friendly encouragement / nudge
            logger.info(
                "Candidate silent for %.1fs. Providing gentle encouragement nudge.",
                candidate_response_timeout,
            )
            is_watchdog_speaking = True
            try:
                if is_english:
                    nudge_text = (
                        "Take your time! Whenever you're ready, feel free to answer, "
                        "or let me know if you'd like to move on to the next question."
                    )
                    await agent_session.say(nudge_text)
                else:
                    await agent_session.generate_reply(
                        instructions=(
                            f"The candidate has been quiet for a moment. In 1 warm, natural, friendly spoken sentence in {interview_language}, "
                            "encourage them to take their time, or offer to move forward if they'd like."
                        )
                    )
            finally:
                is_watchdog_speaking = False

            # Stage 2: Wait for second window before auto-advancing
            remaining_seconds = max(5.0, candidate_max_silence_timeout - candidate_response_timeout)
            logger.info(
                "Silence watchdog waiting additional %.1fs before advancing to next question...",
                remaining_seconds,
            )
            await asyncio.sleep(remaining_seconds)

            if session_done.is_set():
                return
            if agent_session.user_state == "speaking":
                logger.info("Candidate started speaking; auto-advance aborted.")
                return

            # Stage 2: Gracefully advance to next question
            logger.info(
                "Candidate silent for total %.1fs. Auto-advancing smoothly to next question.",
                candidate_max_silence_timeout,
            )
            is_watchdog_speaking = True
            try:
                await agent_session.generate_reply(
                    instructions=(
                        f"The candidate remained silent and was unable to answer the question in time. "
                        f"In a warm, supportive, kind, and encouraging tone in {interview_language}, kindly say no worries at all, "
                        f"and transition smoothly to the next question according to the 5-stage interview plan."
                    )
                )
            finally:
                is_watchdog_speaking = False

        except asyncio.CancelledError:
            logger.debug("Silence watchdog task cancelled.")
        except Exception as e:
            logger.warning("Error in silence watchdog: %s", e)

    @function_tool(description="Conclude the mock interview session once all 5 stages are complete.")
    async def finish_interview(closing_note: str = "") -> str:
        logger.info(f"Concluding interview for {candidate_name}: {closing_note}")
        cancel_silence_watchdog("interview completed")
        await send_transcript_turn(candidate_id, room_name, "interviewer", "Thank you! Your mock interview is now complete.")
        try:
            client = get_http_client()
            await client.post(f"{BACKEND_API_URL}/session/{room_name}/evaluate")
        except Exception as e:
            logger.debug(f"Could not trigger session evaluation: {e}")
        session_done.set()
        return "Interview completed. Candidate can now review transcript and feedback."

    # Generate role, JD, level, language, and resume-driven instructions
    instructions = build_interview_prompt(candidate)

    # Initialize components (STT, TTS, LLM, VAD) calibrated for the interview language
    components = create_pipeline_components(language=interview_language)

    # Pass components to BOTH Agent and AgentSession to guarantee speech generation & LLM replies
    agent = voice.Agent(
        instructions=instructions,
        tools=[finish_interview],
        llm=components["llm"],
        tts=components["tts"],
        stt=components["stt"],
        vad=components.get("vad"),
        allow_interruptions=True,
    )

    agent_session = voice.AgentSession(
        llm=components["llm"],
        tts=components["tts"],
        stt=components["stt"],
        vad=components.get("vad"),
    )

    @agent_session.on("agent_state_changed")
    def on_agent_state_changed(event: voice.AgentStateChangedEvent):
        logger.debug(f"Interviewer state changed: {event.old_state} -> {event.new_state}")
        if event.new_state == "listening":
            if not is_watchdog_speaking and interview_active and not session_done.is_set():
                start_silence_watchdog()
        elif event.new_state in ("thinking", "speaking"):
            if not is_watchdog_speaking:
                cancel_silence_watchdog(f"interviewer transitioned to {event.new_state}")

    @agent_session.on("user_state_changed")
    def on_user_state_changed(event: voice.UserStateChangedEvent):
        logger.debug(f"Candidate state changed: {event.old_state} -> {event.new_state}")
        if event.new_state == "speaking":
            cancel_silence_watchdog("candidate speaking")

    @agent_session.on("conversation_item_added")
    def on_conversation_item_added(event: voice.ConversationItemAddedEvent):
        msg = getattr(event, "item", None)
        if not msg:
            return
        role = getattr(msg, "role", "")
        if role == "system":
            return
        text = (getattr(msg, "raw_text_content", "") or getattr(msg, "text_content", "") or "").strip()
        if not text:
            return
        speaker = "interviewer" if role == "assistant" else "candidate"
        if speaker == "candidate":
            cancel_silence_watchdog("candidate speech turn recorded")
        name_tag = "AI Interviewer" if speaker == "interviewer" else f"Candidate ({candidate_name})"
        logger.info(f"{name_tag}: {text}")
        asyncio.create_task(send_transcript_turn(candidate_id, room_name, speaker, text))

    @agent_session.on("close")
    def on_session_close(event: voice.CloseEvent):
        logger.info(f"Session closed for {candidate_name} ({room_name})")
        cancel_silence_watchdog("session closed")
        session_done.set()
        async def _trigger():
            try:
                client = get_http_client()
                await client.post(f"{BACKEND_API_URL}/session/{room_name}/evaluate")
            except Exception as e:
                logger.debug(f"Could not trigger session evaluation on close: {e}")
        asyncio.create_task(_trigger())

    @ctx.room.on("disconnected")
    def on_room_disconnected():
        logger.info(f"Room '{room_name}' disconnected.")
        cancel_silence_watchdog("room disconnected")
        session_done.set()

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(p: rtc.RemoteParticipant):
        if not p.identity.startswith("agent-"):
            logger.info(f"Candidate participant {p.identity} left room '{room_name}'. Concluding session.")
            cancel_silence_watchdog("candidate disconnected")
            session_done.set()

    # Start voice session inside LiveKit room (handles room connection cleanly without FFI race)
    await agent_session.start(agent, room=ctx.room)

    try:
        await ctx.room.local_participant.set_name("AI Interviewer (Local Dev)")
        logger.info("Set local participant name to 'AI Interviewer (Local Dev)'")
    except Exception as e:
        logger.debug(f"Could not set participant name: {e}")

    logger.info("================================================================")
    logger.info(">>> [AGENT DISPATCHED] Agent '%s' successfully connected to room: '%s'", AGENT_NAME, room_name)
    logger.info("================================================================")

    # Notify backend API that the agent worker is actively running and has joined the room
    try:
        client = get_http_client()
        await client.post(
            f"{BACKEND_API_URL}/interview/agent-status",
            json={
                "room_name": room_name,
                "agent_name": AGENT_NAME,
                "status": "connected",
                "detail": f"Agent worker process started and joined room '{room_name}'",
            },
        )
    except Exception as e:
        logger.debug(f"Could not report agent status to backend: {e}")

    # Check if candidate participant is connected
    has_candidate = any(not p.identity.startswith("agent-") for p in ctx.room.remote_participants.values())
    if not has_candidate:
        logger.info(f"Waiting for candidate to join room '{room_name}'...")
        cand_joined = asyncio.Event()

        @ctx.room.on("participant_connected")
        def on_participant_connected(p: rtc.RemoteParticipant):
            nonlocal candidate_name, candidate, candidate_id
            if not p.identity.startswith("agent-"):
                logger.info(f"Candidate participant {p.identity} joined room (name: {p.name})!")
                if p.name and p.name.strip() and candidate_name in ("Candidate", None, "default_candidate"):
                    candidate_name = p.name.strip()

                async def _load_participant_profile():
                    nonlocal candidate, candidate_name, candidate_id
                    try:
                        p_cand = await fetch_candidate_profile(p.identity)
                        if p_cand and p_cand.get("skills"):
                            merged = dict(p_cand)
                            merged["role"] = interview_role or merged.get("role") or "Software Engineer"
                            merged["position_level"] = interview_level or merged.get("position_level") or "Mid-level"
                            merged["language"] = interview_language or merged.get("language") or "English"
                            merged["job_description"] = interview_jd or merged.get("job_description") or ""
                            candidate = merged
                            candidate_name = candidate.get("name", candidate_name)
                            candidate_id = candidate.get("candidate_id", candidate_id)
                            await agent.update_instructions(build_interview_prompt(candidate))
                            logger.info(f"Updated interviewer instructions for {candidate_name} from participant profile")
                    except Exception as e:
                        logger.debug(f"Error refreshing candidate profile on join: {e}")

                asyncio.create_task(_load_participant_profile())
                cand_joined.set()

        try:
            await asyncio.wait_for(cand_joined.wait(), timeout=90.0)
        except asyncio.TimeoutError:
            logger.info("Candidate wait timeout, proceeding with opening statement.")
    else:
        for p in ctx.room.remote_participants.values():
            if not p.identity.startswith("agent-"):
                if p.name and p.name.strip() and candidate_name in ("Candidate", None, "default_candidate"):
                    candidate_name = p.name.strip()
                p_cand = await fetch_candidate_profile(p.identity)
                if p_cand and p_cand.get("skills"):
                    merged = dict(p_cand)
                    merged["role"] = interview_role or merged.get("role") or "Software Engineer"
                    merged["position_level"] = interview_level or merged.get("position_level") or "Mid-level"
                    merged["language"] = interview_language or merged.get("language") or "English"
                    merged["job_description"] = interview_jd or merged.get("job_description") or ""
                    candidate = merged
                    candidate_name = candidate.get("name", candidate_name)
                    candidate_id = candidate.get("candidate_id", candidate_id)
                    await agent.update_instructions(build_interview_prompt(candidate))

    await asyncio.sleep(1.0)

    primary_skill = (candidate.get("skills") or ["software development"])[0]
    try:
        if is_english:
            opening_greeting = (
                f"Hi {candidate_name}! Welcome to your interview for the {interview_role} role. "
                f"I saw your experience with {primary_skill} on your resume. "
                "To get started, can you tell me a little about yourself and the kind of work you've been doing?"
            )
            logger.info(f"AI Interviewer greeting: {opening_greeting}")
            await agent_session.say(opening_greeting)
            await send_transcript_turn(candidate_id, room_name, "interviewer", opening_greeting)
        else:
            logger.info(f"Generating multilingual opening greeting in {interview_language}...")
            await agent_session.generate_reply(
                instructions=(
                    f"Greet {candidate_name} warmly in simple, everyday, friendly {interview_language} for their {interview_role} interview, "
                    f"mentioning their background in {primary_skill}, and ask them to briefly tell you about themselves in 2 short spoken sentences."
                )
            )
    except Exception as e:
        logger.warning(f"Could not deliver opening greeting: {e}")

    # Activate candidate silence watchdog once opening statement has been delivered
    interview_active = True
    start_silence_watchdog()

    # CRITICAL: Keep interview session running actively until candidate finishes or disconnects
    logger.info(f"AI Interviewer is live and listening in room '{room_name}'. Waiting for interview completion...")
    await session_done.wait()
    cancel_silence_watchdog("session done")
    logger.info(f"Interview session finished for room '{room_name}'. Cleaning up...")


server = AgentServer(
    ws_url=os.getenv("LIVEKIT_URL"),
    api_key=os.getenv("LIVEKIT_API_KEY"),
    api_secret=os.getenv("LIVEKIT_API_SECRET"),
)


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    room_name = getattr(ctx.room, "name", None) or getattr(ctx.job.room, "name", "mock-interview")
    logger.info("================================================================")
    logger.info(">>> [AGENT SERVER STARTED] Agent worker active and picked up room: '%s'", room_name)
    logger.info(">>> Agent Name Registered: '%s'", AGENT_NAME)
    logger.info("================================================================")

    try:
        await run_interview_agent(ctx)
    finally:
        try:
            client = get_http_client()
            await client.post(
                f"{BACKEND_API_URL}/interview/agent-status",
                json={
                    "room_name": room_name,
                    "agent_name": AGENT_NAME,
                    "status": "disconnected",
                    "detail": f"Agent worker exited room '{room_name}'",
                },
            )
        except Exception:
            pass



if __name__ == "__main__":
    cli.run_app(server)
