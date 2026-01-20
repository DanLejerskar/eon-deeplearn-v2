"""
EON DeepLearn™ Script Generator
Generates full-length educational scripts based on target duration.
Uses 150 words per minute for timing calculations.
"""

import os
import json
import re
from datetime import datetime

# Try to import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ScriptGenerator:
    """
    Generates comprehensive educational scripts for video production.
    Target: 150 words per minute of final video.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        self.client = None

        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("[ScriptGen] Anthropic client initialized ✓")
        else:
            print("[ScriptGen] Anthropic not available, will use basic generation")

    def generate_script(self, topic: str, duration_hours: float = 1.0,
                       style: str = "educational", progress_callback=None) -> dict:
        """
        Generate a full-length educational script.

        Args:
            topic: The subject matter
            duration_hours: Target video length in hours
            style: Script style (educational, documentary, tutorial)
            progress_callback: Optional callback for progress updates

        Returns:
            Script dictionary with modules and content
        """
        print(f"[ScriptGen] Generating script for: {topic}")
        print(f"[ScriptGen] Target duration: {duration_hours} hours")

        # Calculate target word count (150 words per minute)
        target_words = int(duration_hours * 60 * 150)
        print(f"[ScriptGen] Target word count: {target_words:,} words")

        if progress_callback:
            progress_callback(10)

        # Generate with Claude if available
        if self.client:
            script = self._generate_with_claude(topic, duration_hours, target_words, style, progress_callback)
        else:
            script = self._generate_basic_script(topic, duration_hours, target_words, style)

        if progress_callback:
            progress_callback(100)

        # Add metadata
        script['target_duration_hours'] = duration_hours
        script['target_words'] = target_words
        script['generated_at'] = datetime.now().isoformat()

        # Count actual words
        actual_words = self._count_words(script)
        script['actual_words'] = actual_words
        print(f"[ScriptGen] Generated {actual_words:,} words ({actual_words/target_words*100:.1f}% of target)")

        return script

    def _generate_with_claude(self, topic: str, duration_hours: float,
                              target_words: int, style: str, progress_callback) -> dict:
        """Generate script using Claude API."""
        print("[ScriptGen] Using Claude API for generation...")

        # Calculate structure
        num_modules = max(5, int(duration_hours * 4))  # ~4 modules per hour
        words_per_module = target_words // num_modules

        if progress_callback:
            progress_callback(20)

        # Generate outline first
        outline_prompt = f"""Create a detailed outline for a {duration_hours}-hour educational video about "{topic}".

Structure it into {num_modules} main modules. For each module, provide:
1. Module title (descriptive, not "Module 1")
2. 3-5 sections within that module
3. Key concepts to cover

Format as JSON:
{{
    "title": "Full video title",
    "modules": [
        {{
            "module": "Module Title Here",
            "sections": [
                {{"title": "Section Title", "key_points": ["point1", "point2", "point3"]}}
            ]
        }}
    ]
}}

Make it comprehensive and educational. The content should flow naturally like a documentary."""

        try:
            outline_response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": outline_prompt}]
            )

            outline_text = outline_response.content[0].text
            outline = self._parse_json_response(outline_text)

            if not outline or 'modules' not in outline:
                print("[ScriptGen] Failed to parse outline, using basic structure")
                return self._generate_basic_script(topic, duration_hours, target_words, style)

        except Exception as e:
            print(f"[ScriptGen] Outline generation error: {e}")
            return self._generate_basic_script(topic, duration_hours, target_words, style)

        if progress_callback:
            progress_callback(40)

        # Now generate full content for each module
        modules_with_content = []
        num_modules = len(outline.get('modules', []))

        for i, module in enumerate(outline.get('modules', [])):
            if progress_callback:
                progress_callback(40 + int(50 * (i + 1) / num_modules))

            module_title = module.get('module', f'Part {i+1}')
            sections = module.get('sections', [])

            # Build section descriptions for the prompt
            section_descriptions = "\n".join([
                f"- {s.get('title', 'Section')}: Cover {', '.join(s.get('key_points', ['key concepts']))}"
                for s in sections
            ])

            content_prompt = f"""Write the full narration script for this module of an educational video about "{topic}".

Module: {module_title}

Sections to cover:
{section_descriptions}

Requirements:
- Write approximately {words_per_module} words of natural, flowing narration
- DO NOT say "Chapter 1" or "Section 1" or "Module 1" - flow naturally between topics
- Use a calm, authoritative documentary tone (like David Attenborough or a BBC documentary)
- Include interesting facts, examples, and explanations
- Make smooth transitions between sections
- Write as continuous prose that will be read aloud
- Do not include any stage directions or [brackets]
- Do not start with "Welcome to" or similar - just begin with the content

Write the complete narration now:"""

            try:
                content_response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8000,
                    messages=[{"role": "user", "content": content_prompt}]
                )

                module_content = content_response.content[0].text.strip()

                modules_with_content.append({
                    "module": module_title,
                    "sections": sections,
                    "content": module_content
                })

                print(f"[ScriptGen] Module {i+1}/{num_modules}: {len(module_content.split())} words ✓")

            except Exception as e:
                print(f"[ScriptGen] Module {i+1} error: {e}")
                # Add placeholder content
                modules_with_content.append({
                    "module": module_title,
                    "sections": sections,
                    "content": f"This section covers {module_title}. " * 50
                })

        return {
            "title": outline.get('title', f"Complete Guide to {topic}"),
            "modules": modules_with_content
        }

    def _generate_basic_script(self, topic: str, duration_hours: float,
                               target_words: int, style: str) -> dict:
        """Generate a basic script without Claude API."""
        print("[ScriptGen] Using basic script generation...")

        num_modules = max(5, int(duration_hours * 4))
        words_per_module = target_words // num_modules

        modules = []
        module_topics = [
            f"Introduction to {topic}",
            f"Fundamental Concepts of {topic}",
            f"Core Principles and Theories",
            f"Practical Applications",
            f"Advanced Topics in {topic}",
            f"Real-World Examples",
            f"Best Practices and Guidelines",
            f"Future Developments",
            f"Summary and Key Takeaways",
            f"Conclusion and Next Steps"
        ]

        for i in range(num_modules):
            module_title = module_topics[i % len(module_topics)]

            # Generate filler content
            sentences = [
                f"Let us explore the fascinating world of {topic}.",
                f"Understanding {topic} requires careful consideration of multiple factors.",
                f"Experts in the field have long studied the implications of {topic}.",
                f"The history of {topic} dates back many years and continues to evolve.",
                f"When examining {topic}, we must consider both theoretical and practical aspects.",
                f"Research has shown that {topic} plays a crucial role in modern understanding.",
                f"The applications of {topic} extend across various domains and disciplines.",
                f"As we delve deeper into {topic}, new insights continue to emerge.",
            ]

            # Repeat to reach word count
            content_parts = []
            current_words = 0
            while current_words < words_per_module:
                for sentence in sentences:
                    content_parts.append(sentence)
                    current_words += len(sentence.split())
                    if current_words >= words_per_module:
                        break

            modules.append({
                "module": module_title,
                "sections": [{"title": "Overview", "key_points": [topic]}],
                "content": " ".join(content_parts)
            })

        return {
            "title": f"Complete Guide to {topic}",
            "modules": modules
        }

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from Claude's response."""
        try:
            # Try direct parse
            return json.loads(text)
        except:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        return None

    def _count_words(self, script: dict) -> int:
        """Count total words in script."""
        total = 0

        # Count title
        total += len(script.get('title', '').split())

        # Count module content
        for module in script.get('modules', []):
            content = module.get('content', '')
            total += len(content.split())

            # Count section content if present
            for section in module.get('sections', []):
                section_content = section.get('content', '')
                total += len(section_content.split())

        return total
