"""
Script Generator Service
Uses Claude API to generate educational scripts
"""

import os
import time
from anthropic import Anthropic

class ScriptGenerator:
    def __init__(self):
        self.client = None
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            self.client = Anthropic(api_key=api_key)

    def generate_outline(self, title: str, content: str = "") -> list:
        """Generate a course outline from title or content"""

        if not self.client:
            # Return mock outline if no API key
            return self._mock_outline(title)

        prompt = f"""Create a detailed course outline for an educational video about: "{title}"

{f'Additional context/content: {content[:2000]}' if content else ''}

Return a JSON array of modules, each with sections. Format:
[
  {{
    "module": "Module 1: Introduction",
    "duration_minutes": 15,
    "sections": [
      {{"title": "Section title", "key_points": ["point 1", "point 2"]}}
    ]
  }}
]

Create 5-8 modules that would make a comprehensive course. Be specific and educational."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            text = response.content[0].text
            # Extract JSON from response
            import json
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return self._mock_outline(title)

        except Exception as e:
            print(f"Error generating outline: {e}")
            return self._mock_outline(title)

    def _mock_outline(self, title: str) -> list:
        """Generate a mock outline for demo purposes"""
        return [
            {
                "module": "Module 1: Introduction & Foundations",
                "duration_minutes": 20,
                "sections": [
                    {"title": "Welcome & Course Overview", "key_points": ["What you'll learn", "Why this matters"]},
                    {"title": "Core Concepts", "key_points": ["Fundamental principles", "Key terminology"]}
                ]
            },
            {
                "module": "Module 2: Deep Dive into Fundamentals",
                "duration_minutes": 25,
                "sections": [
                    {"title": "Understanding the Basics", "key_points": ["Step-by-step breakdown", "Common misconceptions"]},
                    {"title": "Practical Applications", "key_points": ["Real-world examples", "Case studies"]}
                ]
            },
            {
                "module": "Module 3: Advanced Techniques",
                "duration_minutes": 30,
                "sections": [
                    {"title": "Building on Foundations", "key_points": ["Advanced strategies", "Expert insights"]},
                    {"title": "Tools & Resources", "key_points": ["Recommended tools", "Best practices"]}
                ]
            },
            {
                "module": "Module 4: Hands-On Practice",
                "duration_minutes": 25,
                "sections": [
                    {"title": "Guided Exercises", "key_points": ["Practice scenarios", "Step-by-step walkthroughs"]},
                    {"title": "Common Challenges", "key_points": ["Troubleshooting tips", "Solutions to problems"]}
                ]
            },
            {
                "module": "Module 5: Summary & Next Steps",
                "duration_minutes": 20,
                "sections": [
                    {"title": "Key Takeaways", "key_points": ["Review of main concepts", "Action items"]},
                    {"title": "Continuing Your Journey", "key_points": ["Additional resources", "Community & support"]}
                ]
            }
        ]

    def generate_full_script(self, title: str, outline: list, style: str,
                            target_hours: int, job_id: str,
                            progress_callback=None) -> dict:
        """Generate the full script based on outline"""

        # Calculate target word count (150 words per minute)
        target_words = target_hours * 60 * 150
        words_per_module = target_words // len(outline)

        script = {
            "title": title,
            "target_duration_hours": target_hours,
            "style": style,
            "modules": []
        }

        total_modules = len(outline)

        for i, module_outline in enumerate(outline):
            if progress_callback:
                progress_callback(int((i / total_modules) * 100))

            module_script = self._generate_module_script(
                module_outline,
                style,
                words_per_module
            )
            script["modules"].append(module_script)

            # Small delay to show progress
            time.sleep(0.5)

        if progress_callback:
            progress_callback(100)

        return script

    def _generate_module_script(self, module_outline: dict, style: str, target_words: int) -> dict:
        """Generate script for a single module"""

        if not self.client:
            return self._mock_module_script(module_outline, target_words)

        style_instructions = {
            "educational": "Use a clear, informative tone. Focus on teaching concepts thoroughly.",
            "energetic": "Be enthusiastic and engaging! Use dynamic language and exciting examples.",
            "calm": "Use a soothing, relaxed tone. Perfect for bedtime learning or meditation-style content.",
            "corporate": "Professional and business-focused. Use industry terminology appropriately.",
            "documentary": "Narrative storytelling style. Paint vivid pictures with words."
        }

        prompt = f"""Write a detailed script for this educational video module:

Module: {module_outline['module']}
Sections: {module_outline.get('sections', [])}
Target length: approximately {target_words} words
Style: {style_instructions.get(style, style_instructions['educational'])}

Write engaging, educational content. Include:
- Smooth transitions between sections
- Examples and analogies
- Key takeaways

Format as natural spoken narration (no stage directions, just the spoken words)."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                "module": module_outline['module'],
                "content": response.content[0].text,
                "word_count": len(response.content[0].text.split())
            }

        except Exception as e:
            print(f"Error generating module script: {e}")
            return self._mock_module_script(module_outline, target_words)

    def _mock_module_script(self, module_outline: dict, target_words: int) -> dict:
        """Generate mock script content for demo"""

        mock_content = f"""Welcome to {module_outline['module']}.

In this section, we're going to explore some fascinating concepts that will transform your understanding of this subject. Whether you're a complete beginner or looking to deepen your existing knowledge, this module has something valuable for you.

Let's begin by laying the foundation. Understanding the basics is crucial because everything we'll learn later builds upon these core principles. Think of it like constructing a building - without a solid foundation, even the most beautiful structure will eventually crumble.

As we progress through this module, you'll discover practical techniques that you can apply immediately. We'll look at real-world examples, examine case studies, and break down complex ideas into manageable pieces.

One of the most important things to remember is that learning is a journey, not a destination. Take your time with these concepts. Pause if you need to. Revisit sections that challenge you. The goal isn't to rush through - it's to truly understand and internalize this material.

By the end of this module, you'll have a clear grasp of the fundamentals and be ready to tackle more advanced topics. You'll see how these concepts connect to the bigger picture and understand why they matter in practical applications.

Let's dive in and start this exciting learning journey together."""

        # Repeat content to approximate target word count (simplified for demo)
        current_words = len(mock_content.split())
        while current_words < target_words * 0.3:  # Reduced for demo speed
            mock_content += "\n\n" + mock_content
            current_words = len(mock_content.split())

        return {
            "module": module_outline['module'],
            "content": mock_content[:target_words * 6],  # Approximate char limit
            "word_count": len(mock_content.split())
        }
