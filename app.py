"""
EON DeepLearn™ - Web Application
Long-form Educational Video Generation Platform
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import threading

# Import our services
from services.script_generator import ScriptGenerator
from services.video_generator import VideoGenerator
from services.job_manager import JobManager

app = Flask(__name__)
CORS(app)

# Initialize services
script_generator = ScriptGenerator()
video_generator = VideoGenerator()
job_manager = JobManager()

# ============================================================
# ROUTES - Pages
# ============================================================

@app.route('/')
def index():
    """Landing/Dashboard page"""
    return render_template('index.html')

@app.route('/create')
def create():
    """Create new video wizard"""
    return render_template('create.html')

@app.route('/progress/<job_id>')
def progress(job_id):
    """Progress tracking page"""
    return render_template('progress.html', job_id=job_id)

@app.route('/videos')
def videos():
    """List of generated videos"""
    return render_template('videos.html')

# ============================================================
# API ROUTES
# ============================================================

@app.route('/api/generate-outline', methods=['POST'])
def api_generate_outline():
    """Generate course outline from title or content"""
    data = request.json
    title = data.get('title', '')
    content = data.get('content', '')

    try:
        outline = script_generator.generate_outline(title, content)
        return jsonify({'success': True, 'outline': outline})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start-generation', methods=['POST'])
def api_start_generation():
    """Start the video generation process"""
    data = request.json

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = {
        'id': job_id,
        'title': data.get('title', 'Untitled'),
        'outline': data.get('outline', []),
        'voice_style': data.get('voice_style', 'british_documentary'),
        'content_style': data.get('content_style', 'educational'),
        'duration_hours': data.get('duration_hours', 1),
        'image_style': data.get('image_style', 'professional'),
        'image_frequency': data.get('image_frequency', 60),
        'status': 'starting',
        'progress': 0,
        'current_stage': 'Initializing...',
        'created_at': datetime.now().isoformat(),
        'stages': {
            'outline': {'status': 'completed', 'progress': 100},
            'script': {'status': 'pending', 'progress': 0},
            'voice': {'status': 'pending', 'progress': 0},
            'visuals': {'status': 'pending', 'progress': 0},
            'assembly': {'status': 'pending', 'progress': 0}
        }
    }

    job_manager.save_job(job)

    # Start generation in background thread
    thread = threading.Thread(target=run_generation, args=(job_id, data))
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'job_id': job_id})

@app.route('/api/job/<job_id>', methods=['GET'])
def api_get_job(job_id):
    """Get job status"""
    job = job_manager.get_job(job_id)
    if job:
        return jsonify({'success': True, 'job': job})
    return jsonify({'success': False, 'error': 'Job not found'}), 404

@app.route('/api/jobs', methods=['GET'])
def api_list_jobs():
    """List all jobs"""
    jobs = job_manager.list_jobs()
    return jsonify({'success': True, 'jobs': jobs})

@app.route('/api/download/<job_id>')
def api_download(job_id):
    """Download generated video"""
    job = job_manager.get_job(job_id)
    if job and job.get('output_file'):
        directory = os.path.dirname(job['output_file'])
        filename = os.path.basename(job['output_file'])
        return send_from_directory(directory, filename, as_attachment=True)
    return jsonify({'success': False, 'error': 'File not found'}), 404

# ============================================================
# Background Generation
# ============================================================

def run_generation(job_id, data):
    """Run the full generation pipeline in background"""
    try:
        job = job_manager.get_job(job_id)

        # Stage 1: Generate Script
        job_manager.update_job(job_id, {
            'status': 'generating',
            'current_stage': 'Generating script...',
            'progress': 10,
            'stages': {**job['stages'], 'script': {'status': 'in_progress', 'progress': 0}}
        })

        script = script_generator.generate_full_script(
            title=data.get('title'),
            outline=data.get('outline'),
            style=data.get('content_style'),
            target_hours=data.get('duration_hours', 1),
            job_id=job_id,
            progress_callback=lambda p: job_manager.update_job(job_id, {
                'progress': 10 + int(p * 0.3),
                'stages': {**job_manager.get_job(job_id)['stages'], 'script': {'status': 'in_progress', 'progress': p}}
            })
        )

        job_manager.update_job(job_id, {
            'progress': 40,
            'current_stage': 'Script complete! Generating voice...',
            'stages': {**job_manager.get_job(job_id)['stages'],
                      'script': {'status': 'completed', 'progress': 100},
                      'voice': {'status': 'in_progress', 'progress': 0}}
        })

        # Stage 2: Generate Voice (Mock for now)
        voice_file = video_generator.generate_voice(
            script=script,
            voice_style=data.get('voice_style'),
            job_id=job_id,
            progress_callback=lambda p: job_manager.update_job(job_id, {
                'progress': 40 + int(p * 0.25),
                'stages': {**job_manager.get_job(job_id)['stages'], 'voice': {'status': 'in_progress', 'progress': p}}
            })
        )

        job_manager.update_job(job_id, {
            'progress': 65,
            'current_stage': 'Voice complete! Creating visuals...',
            'stages': {**job_manager.get_job(job_id)['stages'],
                      'voice': {'status': 'completed', 'progress': 100},
                      'visuals': {'status': 'in_progress', 'progress': 0}}
        })

        # Stage 3: Generate Visuals (Mock for now)
        visuals = video_generator.generate_visuals(
            script=script,
            style=data.get('image_style'),
            frequency=data.get('image_frequency'),
            job_id=job_id,
            progress_callback=lambda p: job_manager.update_job(job_id, {
                'progress': 65 + int(p * 0.2),
                'stages': {**job_manager.get_job(job_id)['stages'], 'visuals': {'status': 'in_progress', 'progress': p}}
            })
        )

        job_manager.update_job(job_id, {
            'progress': 85,
            'current_stage': 'Visuals complete! Assembling video...',
            'stages': {**job_manager.get_job(job_id)['stages'],
                      'visuals': {'status': 'completed', 'progress': 100},
                      'assembly': {'status': 'in_progress', 'progress': 0}}
        })

        # Stage 4: Assemble Video (Mock for now)
        output_file = video_generator.assemble_video(
            voice_file=voice_file,
            visuals=visuals,
            job_id=job_id,
            progress_callback=lambda p: job_manager.update_job(job_id, {
                'progress': 85 + int(p * 0.15),
                'stages': {**job_manager.get_job(job_id)['stages'], 'assembly': {'status': 'in_progress', 'progress': p}}
            })
        )

        # Complete!
        job_manager.update_job(job_id, {
            'status': 'completed',
            'progress': 100,
            'current_stage': 'Complete!',
            'output_file': output_file,
            'completed_at': datetime.now().isoformat(),
            'stages': {**job_manager.get_job(job_id)['stages'],
                      'assembly': {'status': 'completed', 'progress': 100}}
        })

    except Exception as e:
        job_manager.update_job(job_id, {
            'status': 'failed',
            'current_stage': f'Error: {str(e)}',
            'error': str(e)
        })

# ============================================================
# Run App
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
