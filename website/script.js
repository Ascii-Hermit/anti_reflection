document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    const fileInput = document.getElementById('media-upload');
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        // Send the file to the backend for processing
        fetch('http://localhost:5000/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.message === 'Image processed successfully' || data.message === 'Video processed successfully') {
                if (file.type.startsWith('image')) {
                    // Display the processed image
                    const outputImage = document.getElementById('output-image');
                    outputImage.src = data.processed_image_url;
                    outputImage.style.display = 'block';
                } else if (file.type.startsWith('video')) {
                    // Display the processed video
                    const outputVideo = document.getElementById('output-video');
                    outputVideo.src = data.processed_video_url;
                    outputVideo.style.display = 'block';
                }
            } else {
                alert('Error processing file: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
        });
    }
});
