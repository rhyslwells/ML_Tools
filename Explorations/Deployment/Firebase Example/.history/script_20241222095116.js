// DOM Elements
const form = document.getElementById("countdownForm");
const countdownDisplay = document.getElementById("countdownDisplay");

// Simulated email sending (For local testing only)
function sendEmail(email, heading) {
    alert(`Email sent to ${email} with subject: "${heading}"`);
}

// Start countdown
form.addEventListener("submit", (event) => {
    event.preventDefault();
    const email = document.getElementById("email").value;
    const heading = document.getElementById("heading").value;
    const endTime = new Date(document.getElementById("endTime").value).getTime();

    if (isNaN(endTime)) {
        countdownDisplay.textContent = "Invalid Date and Time.";
        return;
    }

    const interval = setInterval(() => {
        const now = new Date().getTime();
        const timeLeft = endTime - now;

        if (timeLeft <= 0) {
            clearInterval(interval);
            countdownDisplay.textContent = "Time is up!";
            sendEmail(email, heading);
        } else {
            const minutes = Math.floor(timeLeft / (1000 * 60));
            const seconds = Math.floor((timeLeft % (1000 * 60)) / 1000);
            countdownDisplay.textContent = `${minutes}m ${seconds}s remaining.`;
        }
    }, 1000);
});
