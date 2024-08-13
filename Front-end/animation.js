window.addEventListener("DOMContentLoaded", () => {
    // Set speech recognition
    window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    const recognition = new SpeechRecognition(),
          padlock = document.querySelector('.padlock'),
          heardOutput = document.querySelector('.heard-output'),
          openPadlock = () => {
              padlock.classList.add('unlock');
              // Redirect to login.html after unlocking
              setTimeout(() => {
                  window.location.href = 'login.html';
              }, 1000); // Delay for visual effect
          },
          closePadlock = () => {
              padlock.classList.remove('unlock');
          };

    // Start speech recognition
    recognition.start();

    // Listen for when the user finishes talking
    recognition.addEventListener('result', e => {
        // Get transcript of user speech
        const transcript = e.results[0][0].transcript.toLowerCase().replace(/\s/g, '');

        // Output transcript
        heardOutput.textContent = transcript;

        // Check if transcript is valid
        if (transcript === 'helloiamayush' && !padlock.classList.contains('unlock')) {
            openPadlock();
        } else if (transcript === 'lock' && padlock.classList.contains('unlock')) {
            closePadlock();
        }
    });

    // Restart speech recognition after user has finished talking
    recognition.addEventListener('end', recognition.start);

    // Click padlock to open/close
    padlock.addEventListener('click', () => padlock.classList.contains('unlock') ? closePadlock() : openPadlock());
});






