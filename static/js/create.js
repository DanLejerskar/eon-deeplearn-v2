/**
 * EON DeepLearn™ - Create Video Wizard
 */

// State
let currentStep = 1;
let contentSource = null;
let outline = [];
let settings = {
    title: '',
    content: '',
    voice_style: 'british_documentary',
    content_style: 'educational',
    duration_hours: 1,
    image_style: 'professional',
    image_frequency: 60
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initContentOptions();
    initVoiceOptions();
    initStyleOptions();
    initImageStyleOptions();
    initDurationOptions();
    initInputListeners();
});

// Content Source Selection
function initContentOptions() {
    document.querySelectorAll('.content-option').forEach(option => {
        option.addEventListener('click', function() {
            if (this.classList.contains('disabled')) return;

            document.querySelectorAll('.content-option').forEach(o => o.classList.remove('active'));
            this.classList.add('active');

            contentSource = this.dataset.source;

            // Show/hide input fields
            document.getElementById('title-input').style.display = contentSource === 'title' ? 'block' : 'none';
            document.getElementById('content-input').style.display = contentSource === 'content' ? 'block' : 'none';

            updateNextButton();
        });
    });
}

// Voice Selection
function initVoiceOptions() {
    document.querySelectorAll('.voice-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.voice-option').forEach(o => o.classList.remove('active'));
            this.classList.add('active');
            settings.voice_style = this.dataset.voice;
        });
    });
}

// Style Selection
function initStyleOptions() {
    document.querySelectorAll('.style-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.style-option').forEach(o => o.classList.remove('active'));
            this.classList.add('active');
            settings.content_style = this.dataset.style;
        });
    });
}

// Image Style Selection
function initImageStyleOptions() {
    document.querySelectorAll('.image-style-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.image-style-option').forEach(o => o.classList.remove('active'));
            this.classList.add('active');
            settings.image_style = this.dataset.imageStyle;
        });
    });
}

// Duration Selection
function initDurationOptions() {
    document.querySelectorAll('input[name="duration"]').forEach(input => {
        input.addEventListener('change', function() {
            const customDuration = document.getElementById('custom-duration');
            if (this.value === 'custom') {
                customDuration.style.display = 'block';
                settings.duration_hours = parseFloat(document.getElementById('custom-duration-value').value);
            } else {
                customDuration.style.display = 'none';
                settings.duration_hours = parseInt(this.value);
            }
        });
    });

    document.getElementById('custom-duration-value')?.addEventListener('change', function() {
        settings.duration_hours = parseFloat(this.value);
    });

    // Image frequency
    document.getElementById('image-frequency')?.addEventListener('change', function() {
        settings.image_frequency = this.value === 'smart' ? 'smart' : parseInt(this.value);
    });
}

// Input Listeners
function initInputListeners() {
    document.getElementById('video-title')?.addEventListener('input', function() {
        settings.title = this.value;
        updateNextButton();
    });

    document.getElementById('video-title-with-content')?.addEventListener('input', function() {
        settings.title = this.value;
        updateNextButton();
    });

    document.getElementById('video-content')?.addEventListener('input', function() {
        settings.content = this.value;
        updateNextButton();
    });

    document.getElementById('btn-step-1-next')?.addEventListener('click', function() {
        goToStep(2);
    });
}

// Update Next Button State
function updateNextButton() {
    const btn = document.getElementById('btn-step-1-next');
    let isValid = false;

    if (contentSource === 'title' && settings.title.trim().length > 3) {
        isValid = true;
    } else if (contentSource === 'content' && settings.title.trim().length > 3) {
        isValid = true;
    }

    btn.disabled = !isValid;
}

// Navigate to Step
async function goToStep(step) {
    // Hide current step
    document.querySelectorAll('.wizard-panel').forEach(p => p.style.display = 'none');

    // Update progress
    const progress = (step / 5) * 100;
    document.getElementById('wizard-progress-bar').style.width = progress + '%';

    // Update step indicators
    document.querySelectorAll('.wizard-step').forEach((s, i) => {
        s.classList.remove('active', 'completed');
        if (i + 1 < step) s.classList.add('completed');
        if (i + 1 === step) s.classList.add('active');
    });

    // Handle step-specific logic
    if (step === 2) {
        await loadOutline();
    } else if (step === 5) {
        updateSummary();
    }

    // Show new step
    document.getElementById(`step-${step}`).style.display = 'block';
    currentStep = step;

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Load Outline (Step 2)
async function loadOutline() {
    document.getElementById('outline-loading').style.display = 'block';
    document.getElementById('outline-container').style.display = 'none';

    try {
        const response = await fetch('/api/generate-outline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: settings.title,
                content: settings.content
            })
        });

        const data = await response.json();

        if (data.success) {
            outline = data.outline;
            renderOutline();
        } else {
            alert('Error generating outline: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate outline. Please try again.');
    }

    document.getElementById('outline-loading').style.display = 'none';
    document.getElementById('outline-container').style.display = 'block';
}

// Render Outline
function renderOutline() {
    const container = document.getElementById('outline-modules');
    container.innerHTML = '';

    outline.forEach((module, index) => {
        const sections = module.sections || [];
        const sectionsList = sections.map(s =>
            `<div class="section-item">
                <i class="bi bi-check-circle text-success me-2"></i>
                ${s.title}
            </div>`
        ).join('');

        container.innerHTML += `
            <div class="card module-card">
                <div class="card-header d-flex justify-content-between align-items-center"
                     data-bs-toggle="collapse" data-bs-target="#module-${index}">
                    <div>
                        <strong>${module.module}</strong>
                        <small class="text-muted ms-2">${module.duration_minutes || 20} min</small>
                    </div>
                    <i class="bi bi-chevron-down"></i>
                </div>
                <div class="collapse show" id="module-${index}">
                    <div class="card-body">
                        ${sectionsList || '<p class="text-muted mb-0">No sections defined</p>'}
                    </div>
                </div>
            </div>
        `;
    });
}

// Update Summary (Step 5)
function updateSummary() {
    const voiceLabels = {
        'british_documentary': 'British Documentary (Attenborough-style)',
        'american_professional': 'American Professional',
        'warm_teacher': 'Warm & Friendly Teacher'
    };

    const styleLabels = {
        'educational': 'Educational & Factual',
        'energetic': 'Energetic & Captivating',
        'calm': 'Calm & Meditative',
        'documentary': 'Documentary Storytelling'
    };

    const imageLabels = {
        'professional': 'Professional Stock Photos',
        'ai_generated': 'AI Generated Images',
        'infographic': 'Infographics & Diagrams',
        'mixed': 'Mixed (AI Chooses)'
    };

    document.getElementById('summary-title').textContent = settings.title;
    document.getElementById('summary-modules').textContent = outline.length + ' modules';
    document.getElementById('summary-duration').textContent = settings.duration_hours + ' hour(s)';
    document.getElementById('summary-voice').textContent = voiceLabels[settings.voice_style] || settings.voice_style;
    document.getElementById('summary-style').textContent = styleLabels[settings.content_style] || settings.content_style;

    const freq = settings.image_frequency === 'smart' ? 'Smart (AI decides)' : `Every ${settings.image_frequency} seconds`;
    document.getElementById('summary-visuals').textContent =
        (imageLabels[settings.image_style] || settings.image_style) + ', ' + freq;

    // Estimated time based on duration
    const estimatedMinutes = settings.duration_hours * 5 + 5; // ~5 min per hour + overhead
    document.getElementById('estimated-time').textContent = `${estimatedMinutes}-${estimatedMinutes + 10} minutes`;
}

// Start Generation
async function startGeneration() {
    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Starting...';

    try {
        const response = await fetch('/api/start-generation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: settings.title,
                content: settings.content,
                outline: outline,
                voice_style: settings.voice_style,
                content_style: settings.content_style,
                duration_hours: settings.duration_hours,
                image_style: settings.image_style,
                image_frequency: settings.image_frequency
            })
        });

        const data = await response.json();

        if (data.success) {
            // Redirect to progress page
            window.location.href = `/progress/${data.job_id}`;
        } else {
            alert('Error starting generation: ' + data.error);
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-play-fill me-2"></i>Generate Video';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to start generation. Please try again.');
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-fill me-2"></i>Generate Video';
    }
}
