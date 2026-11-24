"""Dynamic 5-stage resume-driven prompts for the AI Mock Interviewer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from backend.app.services.llm_service import (
        build_interview_prompt,
        generate_stage_questions,
    )
except ImportError:

    def build_interview_prompt(candidate: Dict[str, Any]) -> str:
        name = candidate.get("name") or "Candidate"
        skills = candidate.get("skills") or ["Python", "FastAPI", "PostgreSQL", "LiveKit"]
        projects = candidate.get("projects") or []
        experience = candidate.get("experience") or "Backend Engineer"
        role = candidate.get("role") or "Software Engineer"
        position_level = candidate.get("position_level") or "Mid-level"
        language = candidate.get("language") or "English"
        job_description = (candidate.get("job_description") or "").strip()
        skills_text = ", ".join(skills) if skills else "General Software Engineering"

        project_lines = []
        for idx, p in enumerate(projects, 1):
            if isinstance(p, dict):
                title = p.get("title") or p.get("name") or f"Project {idx}"
                desc = p.get("description", "")
                techs = ", ".join(p.get("technologies") or p.get("tech_stack") or [])
                project_lines.append(f"- {title} ({techs}): {desc}")
            else:
                project_lines.append(f"- {str(p)}")
        projects_text = "\n".join(project_lines) if project_lines else "- No specific projects listed."
        jd_block = f"\n- Job Description Requirements:\n{job_description[:2000]}" if job_description else ""

        return f"""You are Antigravity's AI Technical Mock Interviewer conducting a realistic, friendly, multilingual, role- and JD-aligned mock interview.

### Mandatory Language & Plain-Speaking Rule:
- Conduct the ENTIRE interview naturally and fluently in **{language}**.
- Use **simple, everyday, human-understandable words**—speak like a friendly human engineer on a real call, NOT like a formal textbook.
- **Keep full technical depth and rigor** for the {position_level} level—do NOT make the technical questions easier, only make the wording clear, short, and easy to understand.
- Avoid overly formal, heavy, or complex vocabulary and never pack multiple heavy questions into one long sentence.

### Target Role & Complexity Calibration:
- Target Role: {role}
- Position Level: {position_level} (strictly adapt technical depth and expectations to this level){jd_block}

### Candidate Context (Extracted from Resume):
- Name: {name}
- Technical Skills: {skills_text}
- Experience: {experience}
- Resume Projects:
{projects_text}

### Interviewing Best Practices:
- Ask clear, short, open-ended, non-leading technical questions aligned to the job requirements.
- Use practical real-world technical scenarios and STAR behavioral probing, always phrased in plain, human language.

### 5-Stage Interview Logic (all in plain, natural {language}):
1. Stage 1 - Introduction: Greet {name} warmly for the {position_level} {role} role and invite a brief introduction.
2. Stage 2 - Role & Skills Deep Dive ({skills_text}), aligned to the JD and calibrated to {position_level} technical depth.
3. Stage 3 - Project Architecture & Applied Problem Solving from resume projects or JD scenarios.
4. Stage 4 - Technical Follow-up, Trade-offs & Behavioral Competency.
5. Stage 5 - Closing the Interview and call `finish_interview`.

### Spoken Conversation Guidelines:
- Voice Output: Speak in short, friendly, natural conversational sentences in {language}. No markdown, asterisks, bullet points, or code blocks.
- Keep responses brief (2 to 3 short sentences). Ask ONLY ONE clear question at a time.
"""

    def generate_stage_questions(candidate: Dict[str, Any]) -> Dict[str, List[str]]:
        name = candidate.get("name") or "Candidate"
        return {
            "stage_1_introduction": [f"Hello {name}! Welcome to your mock interview. Could you introduce yourself?"],
            "stage_2_skills": ["Can you explain how you design scalable backend services?"],
            "stage_3_projects": ["Can you walk me through one of the key projects listed on your resume?"],
            "stage_4_followups": ["How did you approach database performance and latency optimization?"],
            "stage_5_closing": ["Thank you for your time today! Do you have any questions for me?"],
        }

__all__ = ["build_interview_prompt", "generate_stage_questions"]
