function processImage() {
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/process', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.message === 'Image processed successfully') {
                const imgElement = document.getElementById('processed-image');
                imgElement.src = 'data:image/png;base64,' + data.image_data;
                imgElement.style.display = 'block';
            } else {
                alert('Error processing image.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing image.');
        });
}

function processVideo() {
    const fileInput = document.getElementById('video-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a video to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/process-video-matplotlib', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (response.ok) {
                alert('Matplotlib will display the processed video.');
            } else {
                alert('Error processing video.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing video.');
        });
}

function startRealTimeVideo() {
    fetch('http://127.0.0.1:5000/start-video', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.message === 'Real-time video processing started successfully') {
                alert('Real-time video started.');
                startLiveVideoFeed();
            } else {
                alert('Error starting real-time video.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error starting real-time video.');
        });
}

function stopRealTimeVideo() {
    fetch('http://127.0.0.1:5000/stop-video', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.message === 'Processing stopped successfully') {
                alert('Real-time video stopped.');
                stopLiveVideoFeed();
            } else {
                alert('Error stopping real-time video.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error stopping real-time video.');
        });
}

function startLiveVideoFeed() {
    const imgElement = document.getElementById('live-video');
    imgElement.style.display = 'block';

    const updateFeed = () => {
        fetch('http://127.0.0.1:5000/live-video-frame')
            .then(response => {
                if (response.ok) {
                    return response.blob();
                } else {
                    throw new Error('No content available.');
                }
            })
            .then(blob => {
                const imgUrl = URL.createObjectURL(blob);
                imgElement.src = imgUrl;
            })
            .catch(error => {
                console.error('Error fetching real-time frame:', error);
            });
    };

    window.liveVideoInterval = setInterval(updateFeed, 100);
}

function stopLiveVideoFeed() {
    clearInterval(window.liveVideoInterval);
    const imgElement = document.getElementById('live-video');
    imgElement.style.display = 'none';
}
