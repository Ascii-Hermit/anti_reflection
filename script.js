// Wait for the document to fully load
document.addEventListener('DOMContentLoaded', function () {
    // Set up the form submission handler
    document.getElementById('uploadForm').addEventListener('submit', function(event) {
        event.preventDefault();

        // Get the file input element and create a FormData object
        let formData = new FormData();
        let fileInput = document.getElementById('imageInput');

        // Check if a file is selected
        if (fileInput.files.length === 0) {
            alert('Please select an image or video to upload!');
            return;
        }

        // Append the selected file to the FormData object
        formData.append('file', fileInput.files[0]);

        // Send the file to the Flask backend using a POST request
        fetch('http://127.0.0.1:5000/process', {  // Flask backend endpoint
            method: 'POST',
            body: formData
        })
        .then(response => response.json())  // Parse the JSON response from Flask
        .then(data => {
            // Check if there was a success message in the response
            if (data.message) {
                console.log('Success:', data.message);
                alert(data.message);

                // Optionally, update the webpage with the processed image
                // For example, display a processed image if returned by the server
                if (data.processed_image_url) {
                    // Update the image source to show the result
                    document.getElementById('processedImage').src = data.processed_image_url;
                }
            } else {
                alert('Something went wrong. Please try again.');
            }
        })
        .catch(error => {
            // Catch any errors that occur during the request
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
    });
});
