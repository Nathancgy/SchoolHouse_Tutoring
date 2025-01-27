// Navbar Scroll Effect
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(0, 0, 0, 0.95)';
        navbar.style.padding = '0.5rem 2rem';
    } else {
        navbar.style.background = 'rgba(0, 0, 0, 0.9)';
        navbar.style.padding = '1rem 2rem';
    }
});

// Mobile Menu Toggle
const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('.nav-links');

hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('active');
    hamburger.classList.toggle('active');
});

// Stats Counter Animation
const stats = document.querySelectorAll('.stat-number');

const countUp = (element, target) => {
    let count = 0;
    const speed = 2000 / target;
    
    const updateCount = () => {
        if (count < target) {
            count++;
            element.textContent = count;
            setTimeout(updateCount, speed);
        }
    };
    
    updateCount();
};

// Trigger counter animation when stats section is in view
const observerOptions = {
    threshold: 0.5
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            stats.forEach(stat => {
                const target = parseInt(stat.getAttribute('data-target'));
                countUp(stat, target);
            });
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

const statsSection = document.querySelector('.tournament-stats');
observer.observe(statsSection);

// Live Score Update Animation
const updateScores = () => {
    const scores = document.querySelectorAll('.score');
    scores.forEach(score => {
        const currentScore = parseInt(score.textContent);
        if (Math.random() > 0.7) {
            score.textContent = currentScore + 2;
            score.style.animation = 'pulse 0.5s ease';
            setTimeout(() => {
                score.style.animation = '';
            }, 500);
        }
    });
};

setInterval(updateScores, 3000);

// Add this function at the beginning of the file
function createGameCard(team1, team2, date) {
    return `
        <div class="game-card">
            <div class="game-date">${date}</div>
            <div class="game-teams">
                <div class="team">
                    <img src="https://raw.githubusercontent.com/your-username/your-repo/main/images/${team1.toLowerCase()}.png" 
                         alt="${team1}" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjMmMzZTUwIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjwvc3ZnPg=='">
                    <span>${team1}</span>
                </div>
                <div class="vs-badge">VS</div>
                <div class="team">
                    <img src="https://raw.githubusercontent.com/your-username/your-repo/main/images/${team2.toLowerCase()}.png" 
                         alt="${team2}"
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveZ0iMCAwIDI0IDI0IiBmaWxsPSIjMmMzZTUwIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjwvc3ZnPg=='">
                    <span>${team2}</span>
                </div>
            </div>
            <button class="ticket-button">Book Now</button>
        </div>
    `;
}

// Add this after your existing code
const gamesGrid = document.querySelector('.games-grid');
const upcomingGames = [
    { team1: 'Lakers', team2: 'Celtics', date: 'June 15, 2024' },
    { team1: 'Warriors', team2: 'Bulls', date: 'June 18, 2024' },
    { team1: 'Nets', team2: 'Heat', date: 'June 20, 2024' }
];

// Populate games grid
upcomingGames.forEach(game => {
    gamesGrid.innerHTML += createGameCard(game.team1, game.team2, game.date);
}); 