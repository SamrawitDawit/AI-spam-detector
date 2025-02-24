document.getElementById('spam-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const message = document.getElementById('message').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = data.result;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});