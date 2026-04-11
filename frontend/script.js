// Hugging Face Space URL (update with your actual space URL)
const API_URL = 'https://your-username-student-performance-predictor.hf.space/predict';

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading state
    const btn = document.getElementById('predictBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = 'Predicting... <span class="loading"></span>';
    btn.disabled = true;
    
    // Collect form data
    const formData = {
        hours_studied: parseFloat(document.getElementById('hours_studied').value),
        attendance: parseFloat(document.getElementById('attendance').value),
        sleep_hours: parseFloat(document.getElementById('sleep_hours').value),
        previous_scores: parseFloat(document.getElementById('previous_scores').value),
        tutoring_sessions: parseFloat(document.getElementById('tutoring_sessions').value),
        physical_activity: parseFloat(document.getElementById('physical_activity').value),
        parental_involvement: document.getElementById('parental_involvement').value,
        access_to_resources: document.getElementById('access_to_resources').value,
        extracurricular: document.getElementById('extracurricular').value,
        motivation_level: document.getElementById('motivation_level').value,
        internet_access: document.getElementById('internet_access').value,
        family_income: document.getElementById('family_income').value,
        teacher_quality: document.getElementById('teacher_quality').value,
        school_type: document.getElementById('school_type').value,
        peer_influence: document.getElementById('peer_influence').value,
        learning_disabilities: document.getElementById('learning_disabilities').value,
        parental_education: document.getElementById('parental_education').value,
        gender: document.getElementById('gender').value,
        distance_from_home: document.getElementById('distance_from_home').value
    };
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show result
            document.getElementById('predictionValue').textContent = data.predicted_score;
            document.getElementById('grade').innerHTML = getGradeHTML(data.predicted_score);
            document.getElementById('result').classList.add('show');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error connecting to server: ' + error.message);
    } finally {
        // Reset button
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

function getGradeHTML(score) {
    if (score >= 90) return '🎉 Grade: A (Excellent) 🎉';
    if (score >= 80) return '✅ Grade: B (Very Good) ✅';
    if (score >= 70) return '📘 Grade: C (Good) 📘';
    if (score >= 60) return '⚠️ Grade: D (Below Average) ⚠️';
    return '❌ Grade: F (Failing) ❌';
}