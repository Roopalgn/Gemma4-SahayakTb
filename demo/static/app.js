/* ============================================================
   SAHAYAK-TB — Frontend Logic (Enhanced)
   ============================================================ */

// ---------------------------------------------------------------------------
// Example Data
// ---------------------------------------------------------------------------
const EXAMPLES = [
    // Low risk
    `Visit 1 (Month 1): Priya Sharma, 28F, school teacher. Category I new case. Weight 52kg. Lives 10 minutes walk from DOT center. Husband accompanies her daily. Family fully supportive. No smoking, no alcohol, no comorbidities.

Visit 2 (Month 2): Coming regularly without missing a single day. Weight 53kg. No side effects reported. Husband still accompanying. Maintaining medicine diary. Sputum converted negative.

Visit 3 (Month 3): Perfect adherence continues. Weight 54kg. Understands she must complete full 6 months. No complaints. Family celebrating her progress. School gave her medical leave with full pay.`,

    // Moderate risk
    `Visit 1 (Month 1): Lakshmi Devi, 55F, homemaker. Category I new case. Weight 42kg. Diabetic — on metformin. Son brings her, DOT center about 40 min by auto.

Visit 2 (Month 2): Weight 42kg. Mild nausea. Blood sugar was 220 fasting. Son says his work schedule may change next month — may not be able to bring her.`,

    // High risk
    `Visit 1 (Month 1): Ramesh Kumar, 35M, daily wage construction worker. Category I new case. Weight 52kg. Lives in labour colony, 1.5 hours from DOT center by bus. Bidi peeta hai, advised to quit.

Visit 2 (Month 2): Weight 51kg. Ji machlata hai dawai lene ke baad. Missed 2 days last week because of work. Still smoking.

Visit 3 (Month 3): Ab theek lag raha hai, khansi band ho gayi. Wants to stop early. Bus fare getting expensive — aane me bahut paisa lagta hai.`,

    // Critical risk
    `Visit 1 (Month 1): Suresh Yadav, 42M, migrant construction worker from Bihar. Retreatment case — pehle bhi treatment hua tha 3 saal pehle but left after 3 months. Category II. Sharab peeta hai daily. Temporary shed at construction site. No permanent address.

Visit 2 (Month 2): Missed one week — doosre site pe kaam. Weight dropped 1kg. Nausea and joint pain. Still drinking. No family support here.

Visit 3 (Month 3): Came after missing 4 days. Bahut thak jaata hai. Pet me jalan. Weight down again. Mentioned going to Bihar for family event. Still drinking.

Visit 4 (Month 4): Almost didn't come. Looks thin. Says ab theek hai, wants to stop. Going to Bihar next week. Drinking continues. No sputum test since start.`,

    // Mixed Hindi-English
    `Visit 1 (Month 1): Anil Verma, 22M, college chhod diya TB ki wajah se. Category I. Weight 55kg. Akela rehta hai hostel me. Parents gaon me hain. Bidi peeta hai kabhi kabhi.

Visit 2 (Month 2): Dawai se bahut neend aati hai, class me dhyan nahi lag raha. Ek baar skip kiya dose. Dost nahi jaante bimari ke baare me — sharam aati hai.

Visit 3 (Month 3): Do din late aaya. Ab kaafi theek hai, khansi nahi. Poochha ki kya 4 mahine me chhod sakte hain. Exam aa rahe hain. Paise bhi khatam ho rahe hain.`,
];

// ---------------------------------------------------------------------------
// Scroll Reveal (Intersection Observer)
// ---------------------------------------------------------------------------
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                const delay = getComputedStyle(entry.target).getPropertyValue('--delay');
                if (delay) {
                    entry.target.style.transitionDelay = delay;
                }
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.fade-in-up, .fade-in-left').forEach(el => {
        observer.observe(el);
    });
}

// ---------------------------------------------------------------------------
// Counter Animation
// ---------------------------------------------------------------------------
function initCounters() {
    const counters = document.querySelectorAll('.counter');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.dataset.target);
                animateCounter(el, 0, target, 1500);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(c => observer.observe(c));
}

function animateCounter(el, start, end, duration) {
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(start + (end - start) * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ---------------------------------------------------------------------------
// Scroll Progress Bar
// ---------------------------------------------------------------------------
function initScrollProgress() {
    const progressBar = document.getElementById('scroll-progress');
    if (!progressBar) return;
    
    window.addEventListener('scroll', () => {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        progressBar.style.width = progress + '%';
    }, { passive: true });
}

// ---------------------------------------------------------------------------
// Navbar scroll effect
// ---------------------------------------------------------------------------
function initNavbar() {
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        const scrollY = window.scrollY;
        if (scrollY > 100) {
            navbar.style.background = 'rgba(10, 14, 26, 0.9)';
            navbar.style.backdropFilter = 'blur(24px)';
            navbar.style.boxShadow = '0 4px 30px rgba(0, 0, 0, 0.3)';
        } else {
            navbar.style.background = 'var(--bg-glass)';
            navbar.style.boxShadow = 'none';
        }
    }, { passive: true });
}

// ---------------------------------------------------------------------------
// Active nav link highlight
// ---------------------------------------------------------------------------
function initActiveNav() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === '#' + id);
                });
            }
        });
    }, { threshold: 0.3 });
    
    sections.forEach(s => observer.observe(s));
}

// ---------------------------------------------------------------------------
// Animated Bar Chart (trigger on scroll)
// ---------------------------------------------------------------------------
function initBarCharts() {
    const bars = document.querySelectorAll('.bar');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.width = getComputedStyle(entry.target).getPropertyValue('--bar-width');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    bars.forEach(bar => {
        bar.style.width = '0%';
        observer.observe(bar);
    });
}

// ---------------------------------------------------------------------------
// Eval Ring Animation (trigger on scroll)
// ---------------------------------------------------------------------------
function initEvalRings() {
    const rings = document.querySelectorAll('.eval-ring-progress');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Re-trigger animation by resetting
                entry.target.style.transition = 'none';
                entry.target.style.strokeDashoffset = '326.73';
                requestAnimationFrame(() => {
                    entry.target.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.16, 1, 0.3, 1)';
                    const progress = getComputedStyle(entry.target).getPropertyValue('--progress');
                    entry.target.style.strokeDashoffset = 326.73 - (326.73 * parseFloat(progress) / 100);
                });
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    rings.forEach(r => observer.observe(r));
}

// ---------------------------------------------------------------------------
// Tier Bar Animation (trigger on scroll)
// ---------------------------------------------------------------------------
function initTierBars() {
    const fills = document.querySelectorAll('.tier-bar-fill');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.width = getComputedStyle(entry.target).getPropertyValue('--fill');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    fills.forEach(f => {
        f.style.width = '0%';
        observer.observe(f);
    });
}

// ---------------------------------------------------------------------------
// Load Example
// ---------------------------------------------------------------------------
let _exampleCycleIndex = 0;
function loadExample(index) {
    if (index === undefined) {
        index = _exampleCycleIndex;
        _exampleCycleIndex = (_exampleCycleIndex + 1) % EXAMPLES.length;
    }
    const textarea = document.getElementById('visit-notes');
    textarea.value = EXAMPLES[index];
    textarea.focus();
    textarea.style.borderColor = 'var(--primary)';
    textarea.style.boxShadow = '0 0 20px rgba(8, 145, 178, 0.2)';
    setTimeout(() => { textarea.style.borderColor = ''; textarea.style.boxShadow = ''; }, 1200);
}

// ---------------------------------------------------------------------------
// Assess Risk
// ---------------------------------------------------------------------------
async function assessRisk() {
    const notes = document.getElementById('visit-notes').value.trim();
    if (!notes) {
        shakeElement(document.getElementById('visit-notes'));
        return;
    }

    const btn = document.getElementById('assess-btn');
    btn.classList.add('loading');
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('loading-state').classList.remove('hidden');

    try {
        const response = await fetch('/api/assess', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ visit_notes: notes }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        console.error(err);
        document.getElementById('loading-state').classList.add('hidden');
        document.getElementById('empty-state').classList.remove('hidden');
        document.getElementById('empty-state').innerHTML = `
            <div class="empty-icon">⚠️</div>
            <p style="color: var(--danger); font-weight: 600;">Error: ${escapeHtml(err.message)}</p>
            <p style="font-size: 0.8rem; margin-top: 8px; color: var(--text-muted);">The model may still be loading. Please try again in a moment.</p>
        `;
    } finally {
        btn.classList.remove('loading');
    }
}

// ---------------------------------------------------------------------------
// Render Results
// ---------------------------------------------------------------------------
function renderResults(data) {
    document.getElementById('loading-state').classList.add('hidden');
    document.getElementById('results').classList.remove('hidden');

    // Tier card
    const tierCard = document.getElementById('tier-card');
    const tier = data.dropout_risk_tier || 'unknown';
    tierCard.className = `tier-card glass-card tier-${tier}`;
    document.getElementById('tier-badge').textContent = tier === 'unknown' ? '—' : `${tier} risk`;
    document.getElementById('tier-meta').textContent = 
        `Patient: ${escapeHtml(data.patient_id || '—')} · Visit ${data.visit_number || '?'} · Month ${data.treatment_month || '?'} · Confidence: ${data.confidence || '—'}`;

    // Factors
    const factorsList = document.getElementById('factors-list');
    const factors = data.risk_factors || [];
    if (factors.length === 0) {
        factorsList.innerHTML = '<p class="no-factors">No risk factors identified — patient appears adherent.</p>';
    } else {
        factorsList.innerHTML = factors.map((f, i) => `
            <div class="factor-card" style="animation-delay: ${i * 0.1}s">
                <div class="factor-name">${escapeHtml(f.factor)}</div>
                <div class="factor-detail">
                    <strong>First noted:</strong> Visit ${f.visit_first_noted || '?'}<br>
                    <strong>Evidence:</strong> ${escapeHtml(f.clinical_basis || 'N/A')}<br>
                    <strong>Action:</strong> ${escapeHtml(f.dots_implication || 'N/A')}
                </div>
            </div>
        `).join('');
    }

    // Action
    document.getElementById('action-text').textContent = data.recommended_action || 'No action specified.';

    // Completeness
    document.getElementById('completeness-text').textContent = data.data_completeness || 'Not assessed.';

    // JSON
    document.getElementById('json-output').textContent = JSON.stringify(data, null, 2);

    // Scroll to results on mobile
    if (window.innerWidth < 768) {
        document.getElementById('tier-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function shakeElement(el) {
    el.style.animation = 'shake 0.4s ease-in-out';
    el.addEventListener('animationend', () => { el.style.animation = ''; }, { once: true });
}

// Shake keyframes
const shakeStyle = document.createElement('style');
shakeStyle.textContent = `@keyframes shake { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-6px)} 75%{transform:translateX(6px)} }`;
document.head.appendChild(shakeStyle);

// ---------------------------------------------------------------------------
// Smooth scroll for nav links
// ---------------------------------------------------------------------------
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initScrollReveal();
    initCounters();
    initScrollProgress();
    initNavbar();
    initActiveNav();
    initBarCharts();
    initEvalRings();
    initTierBars();
    initSmoothScroll();
});

// Keyboard shortcut: Ctrl+Enter to submit
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        assessRisk();
    }
});
