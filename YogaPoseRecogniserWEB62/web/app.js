
let model, webcam, ctx, labelContainer, maxPredictions;
const poseImages = new Map();
let currentPoseImage = null;
let currentPoseIndex = 0;
let poseHoldTimer = 3;
let lastPoseTime = 0;
let poseThreshold = 0.8; // Default 80%
const poseOrder = ['Pose1', 'Pose2', 'Pose3', 'Pose4', 'Pose5', 'Pose6'];

document.addEventListener('DOMContentLoaded', () => {
    const savedTimer = localStorage.getItem('pose_timer');
    if (savedTimer) {
        document.getElementById('pose-timer').value = savedTimer;
        poseHoldTimer = parseInt(savedTimer);
    }
    document.getElementById('start-button').addEventListener('click', startRecognition);
    document.getElementById('back-button').addEventListener('click', showSettingsPage);
    document.getElementById('save-settings').addEventListener('click', saveSettings);

    poseOrder.forEach((poseName, index) => {
        document.getElementById(`${poseName.toLowerCase()}-image`)
            .addEventListener('change', (e) => handleImageUpload(e, poseName));
    });

    // Load saved settings
    const savedModelUrl = localStorage.getItem('model_url');
    if (savedModelUrl) {
        document.getElementById('model-url').value = savedModelUrl;
    }
    
    const savedThreshold = localStorage.getItem('pose_threshold');
    if (savedThreshold) {
        document.getElementById('pose-threshold').value = savedThreshold;
        poseThreshold = savedThreshold / 100;
    }

    poseOrder.forEach(poseName => {
        const savedImage = localStorage.getItem(`pose_${poseName}`);
        if (savedImage) {
            const preview = document.getElementById(`${poseName.toLowerCase()}-preview`);
            preview.src = savedImage;
            preview.style.display = 'block';
            poseImages.set(poseName, savedImage);
        }
    });
});

function saveSettings() {
    const modelUrl = document.getElementById('model-url').value;
    const threshold = document.getElementById('pose-threshold').value;
    const timer = document.getElementById('pose-timer').value;
    localStorage.setItem('model_url', modelUrl);
    localStorage.setItem('pose_threshold', threshold);
    localStorage.setItem('pose_timer', timer);
    poseThreshold = threshold / 100;
    poseHoldTimer = parseInt(timer);
    showSaveConfirmation();

    poseOrder.forEach(poseName => {
        const poseImage = poseImages.get(poseName);
        if (poseImage) {
            localStorage.setItem(`pose_${poseName}`, poseImage);
        }
    });
}

function handleImageUpload(event, poseName) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById(`${poseName.toLowerCase()}-preview`);
            const label = event.target.parentElement.querySelector('label');
            preview.src = e.target.result;
            preview.style.display = 'block';
            poseImages.set(poseName, e.target.result);
            label.textContent = `${poseName} Image: ${file.name}`;
        };
        reader.readAsDataURL(file);
    }
}

function showSaveConfirmation() {
    const saveButton = document.getElementById('save-settings');
    const originalText = saveButton.textContent;
    saveButton.textContent = 'Settings Saved!';
    saveButton.style.backgroundColor = '#45a049';
    setTimeout(() => {
        saveButton.textContent = originalText;
        saveButton.style.backgroundColor = '#4CAF50';
    }, 2000);
}

function showSettingsPage() {
    document.getElementById('recognition-page').classList.remove('active');
    document.getElementById('settings-page').classList.add('active');
    if (webcam) {
        webcam.stop();
    }
}

async function startRecognition() {
    if (poseImages.size < 6) {
        alert('Please upload all six pose images first');
        return;
    }

    currentPoseIndex = 0; // Reset to start from pose 1
    document.getElementById('settings-page').classList.remove('active');
    document.getElementById('recognition-page').classList.add('active');

    const URL = document.getElementById('model-url').value;
    await init(URL);
}

async function init(URL) {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    const size = 640;
    const flip = true;
    webcam = new tmPose.Webcam(size, size, flip);
    await webcam.setup();
    await webcam.play();

    const canvas = document.getElementById('output');
    ctx = canvas.getContext('2d');

    canvas.width = size;
    canvas.height = size;

    window.requestAnimationFrame(loop);
}

async function loop(timestamp) {
    webcam.update();
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    const prediction = await model.predict(posenetOutput);

    ctx.drawImage(webcam.canvas, 0, 0);
    if (pose) {
        drawPose(pose);
    }

    let maxConfidence = 0;
    let bestPose = '';
    for (let i = 0; i < maxPredictions; i++) {
        if (prediction[i].probability > maxConfidence) {
            maxConfidence = prediction[i].probability;
            bestPose = prediction[i].className;
        }
    }

    const expectedPose = poseOrder[currentPoseIndex];
    
    // Update pose image
    const poseCompare = document.getElementById('pose-compare');
    poseCompare.src = poseImages.get(expectedPose);
    
    const timerBox = document.getElementById('timer-box');
    const expectedPoseEl = document.getElementById('expected-pose');
    const currentPoseEl = document.getElementById('current-pose-text');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');

    expectedPoseEl.textContent = expectedPose;
    currentPoseEl.textContent = bestPose;
    
    const confidencePercent = (maxConfidence * 100).toFixed(2);
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceText.textContent = `${confidencePercent}%`;

    if (maxConfidence > poseThreshold && bestPose === expectedPose) {
        if (lastPoseTime === 0) {
            lastPoseTime = Date.now();
        }
        const holdTime = poseHoldTimer - Math.floor((Date.now() - lastPoseTime) / 1000);

        if (holdTime <= 0) {
            currentPoseIndex = (currentPoseIndex + 1) % poseOrder.length;
            lastPoseTime = 0;
        } else {
            timerBox.textContent = `${Math.max(0, holdTime)}s`;
            timerBox.style.backgroundColor = '#4CAF50';
        }
    } else {
        lastPoseTime = 0;
        timerBox.textContent = `${poseHoldTimer}s`;
        timerBox.style.backgroundColor = '#ff4444';
    }
}

function drawPose(pose) {
    if (!pose || !pose.keypoints) return;

    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;

    // Draw dots for head keypoints (0-4)
    for (let i = 0; i < 5; i++) {
        const point = pose.keypoints[i];
        if (point.score > 0.3) {
            ctx.beginPath();
            ctx.arc(point.position.x, point.position.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#00ff00';
            ctx.fill();
        }
    }

    const connections = [
        [5, 7], [7, 9],     // Left arm
        [6, 8], [8, 10],    // Right arm
        [5, 6],             // Shoulders
        [5, 11], [6, 12],   // Torso
        [11, 12],           // Hips
        [11, 13], [13, 15], // Left leg
        [12, 14], [14, 16]  // Right leg
    ];

    for (const [i1, i2] of connections) {
        const point1 = pose.keypoints[i1];
        const point2 = pose.keypoints[i2];

        if (point1.score > 0.3 && point2.score > 0.3) {
            ctx.beginPath();
            ctx.moveTo(point1.position.x, point1.position.y);
            ctx.lineTo(point2.position.x, point2.position.y);
            ctx.stroke();
        }
    }
}
