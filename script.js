document.getElementById('upload-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch('/process', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        const outputSection = document.getElementById('output-section');
        const outputContainer = document.getElementById('output-container');

        outputContainer.innerHTML = '';

        if (data.type === 'image') {
            const img = document.createElement('img');
            img.src = data.output_url;
            outputContainer.appendChild(img);
        } else if (data.type === 'video') {
            const video = document.createElement('video');
            video.src = data.output_url;
            video.controls = true;
            outputContainer.appendChild(video);
        }

        outputSection.style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
});
