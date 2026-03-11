document.addEventListener('DOMContentLoaded', () => {
    // Media Type Tabs Logic
    const tabs = document.querySelectorAll('.tab-btn');
    const mediaTypeInput = document.getElementById('media-type');
    const fileInput = document.getElementById('file-input');
    
    // Config extensions
    const acceptMap = {
        'image': '.jpg,.jpeg,.png',
        'video': '.mp4,.avi',
        'audio': '.wav,.mp3'
    };

    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            // Remove active
            tabs.forEach(t => t.classList.remove('active'));
            // Add active
            tab.classList.add('active');
            
            const type = tab.getAttribute('data-type');
            mediaTypeInput.value = type;
            
            // Adjust file input accept constraint
            fileInput.accept = acceptMap[type];
            fileInput.value = ''; // clear
            document.getElementById('file-info').innerText = 'No file selected. Please select a ' + type + ' file.';
            document.getElementById('analyze-btn').disabled = true;
        });
    });

    // File Drop Zone Logic
    const dropZone = document.getElementById('drop-zone');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileInfo = document.getElementById('file-info');

    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });

        fileInput.addEventListener('change', handleFileSelect);

        function handleFileSelect() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                fileInfo.innerHTML = `<strong>Selected:</strong> ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                analyzeBtn.disabled = false;
            } else {
                fileInfo.innerText = "No file selected.";
                analyzeBtn.disabled = true;
            }
        }
    }

    // Form Submission & Analysis Simulation
    const form = document.getElementById('upload-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Switch UI to Loading
            document.getElementById('analyze-btn').classList.add('hidden');
            document.getElementById('loading-state').classList.remove('hidden');
            document.getElementById('loading-text').innerText = "Uploading to Cloud Pipeline...";
            
            document.getElementById('empty-result').classList.add('hidden');
            document.getElementById('result-container').classList.add('hidden');
            
            const formData = new FormData(form);
            
            try {
                // Change text context halfway
                setTimeout(() => {
                    document.getElementById('loading-text').innerText = "Running Deep Learning Inference...";
                }, 1000);

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Hide loading
                document.getElementById('loading-state').classList.add('hidden');
                document.getElementById('analyze-btn').classList.remove('hidden');
                
                if (data.success) {
                    displayResults(data.result);
                } else {
                    alert("Error: " + data.error);
                }
            } catch (error) {
                console.error("Upload error:", error);
                document.getElementById('loading-state').classList.add('hidden');
                document.getElementById('analyze-btn').classList.remove('hidden');
                alert("Failed to connect to the analysis engine. Please try again.");
            }
        });
    }

    function displayResults(result) {
        const resultContainer = document.getElementById('result-container');
        const stamp = document.getElementById('result-label');
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceText = document.getElementById('confidence-text');
        const details = document.getElementById('result-details');
        
        resultContainer.classList.remove('hidden');
        
        // Populate specific data
        stamp.innerText = result.label;
        if (result.label === 'FAKE') {
            stamp.className = 'result-stamp stamp-fake';
            confidenceBar.style.background = `linear-gradient(90deg, #333 40%, #ef4444 100%)`; // Red gradient
            confidenceBar.style.width = `100%`; // Fills up to 100 visually mapping to confidence later
        } else {
            stamp.className = 'result-stamp stamp-real';
            confidenceBar.style.background = `linear-gradient(90deg, #333 40%, #10b981 100%)`; // Green gradient
        }
        
        details.innerText = result.details || "Finished classification through neural network model.";
        
        // Animate counter
        let currentConf = 0;
        const targetConf = result.confidence;
        confidenceBar.style.width = '0%';
        
        setTimeout(() => {
            confidenceBar.style.width = `${targetConf}%`;
        }, 100);
        
        const interval = setInterval(() => {
            currentConf += 2;
            if (currentConf >= targetConf) {
                currentConf = targetConf;
                clearInterval(interval);
            }
            confidenceText.innerText = currentConf.toFixed(1) + "%";
        }, 30);
    }
});
