// Game Variables
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const playerScoreElement = document.getElementById('playerScore');
const aiScoreElement = document.getElementById('aiScore');
const gameStatusElement = document.getElementById('gameStatus');
const gameOverScreen = document.getElementById('gameOverScreen');
const gameOverTitle = document.getElementById('gameOverTitle');
const gameOverMessage = document.getElementById('gameOverMessage');
const restartButton = document.getElementById('restartButton');

// Game State
let gameState = 'menu'; // 'menu', 'playing', 'paused', 'gameOver'
let animationId;
let lastTime = 0;

// Game Objects
const game = {
    width: canvas.width,
    height: canvas.height,
    playerScore: 0,
    aiScore: 0,
    winScore: 10
};

// Ball Object
const ball = {
    x: game.width / 2,
    y: game.height / 2,
    radius: 10,
    speedX: 5,
    speedY: 3,
    maxSpeed: 12,
    color: '#ffffff'
};

// Player Paddle
const playerPaddle = {
    x: 20,
    y: game.height / 2 - 50,
    width: 15,
    height: 100,
    speed: 7,
    color: '#4CAF50'
};

// AI Paddle
const aiPaddle = {
    x: game.width - 35,
    y: game.height / 2 - 50,
    width: 15,
    height: 100,
    speed: 5,
    color: '#f44336'
};

// Input Handling
const keys = {
    w: false,
    s: false,
    space: false,
    enter: false
};

// Event Listeners
document.addEventListener('keydown', (e) => {
    switch(e.key.toLowerCase()) {
        case 'w':
            keys.w = true;
            break;
        case 's':
            keys.s = true;
            break;
        case ' ':
            if (!keys.space) {
                keys.space = true;
                handlePause();
            }
            e.preventDefault();
            break;
        case 'enter':
            if (!keys.enter) {
                keys.enter = true;
                handleRestart();
            }
            e.preventDefault();
            break;
    }
});

document.addEventListener('keyup', (e) => {
    switch(e.key.toLowerCase()) {
        case 'w':
            keys.w = false;
            break;
        case 's':
            keys.s = false;
            break;
        case ' ':
            keys.space = false;
            break;
        case 'enter':
            keys.enter = false;
            break;
    }
});

// Restart button event
restartButton.addEventListener('click', () => {
    resetGame();
    gameState = 'playing';
    updateGameStatus();
    hideGameOver();
    gameLoop();
});

// Game Functions
function resetGame() {
    // Reset ball
    ball.x = game.width / 2;
    ball.y = game.height / 2;
    ball.speedX = Math.random() > 0.5 ? 5 : -5;
    ball.speedY = (Math.random() - 0.5) * 6;
    
    // Reset paddles
    playerPaddle.y = game.height / 2 - playerPaddle.height / 2;
    aiPaddle.y = game.height / 2 - aiPaddle.height / 2;
    
    // Reset scores
    game.playerScore = 0;
    game.aiScore = 0;
    updateScore();
}

function resetBall() {
    ball.x = game.width / 2;
    ball.y = game.height / 2;
    ball.speedX = Math.random() > 0.5 ? 5 : -5;
    ball.speedY = (Math.random() - 0.5) * 6;
}

function updateScore() {
    playerScoreElement.textContent = game.playerScore;
    aiScoreElement.textContent = game.aiScore;
}

function updateGameStatus() {
    switch(gameState) {
        case 'menu':
            gameStatusElement.textContent = 'Press Space to Start';
            break;
        case 'playing':
            gameStatusElement.textContent = 'Playing - Space to Pause';
            break;
        case 'paused':
            gameStatusElement.textContent = 'Paused - Space to Resume';
            break;
        case 'gameOver':
            gameStatusElement.textContent = 'Game Over - Enter to Restart';
            break;
    }
}

function handlePause() {
    if (gameState === 'menu') {
        gameState = 'playing';
        gameLoop();
    } else if (gameState === 'playing') {
        gameState = 'paused';
        cancelAnimationFrame(animationId);
    } else if (gameState === 'paused') {
        gameState = 'playing';
        gameLoop();
    }
    updateGameStatus();
}

function handleRestart() {
    if (gameState === 'gameOver') {
        resetGame();
        gameState = 'playing';
        hideGameOver();
        gameLoop();
    } else if (gameState !== 'menu') {
        resetGame();
        gameState = 'playing';
        hideGameOver();
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        gameLoop();
    }
    updateGameStatus();
}

function showGameOver(winner) {
    gameState = 'gameOver';
    gameOverTitle.textContent = 'Game Over!';
    gameOverMessage.textContent = winner === 'player' ? 'You Win!' : 'AI Wins!';
    gameOverScreen.classList.remove('hidden');
    updateGameStatus();
}

function hideGameOver() {
    gameOverScreen.classList.add('hidden');
}

// Game Logic
function updatePlayerPaddle() {
    if (keys.w && playerPaddle.y > 0) {
        playerPaddle.y -= playerPaddle.speed;
    }
    if (keys.s && playerPaddle.y < game.height - playerPaddle.height) {
        playerPaddle.y += playerPaddle.speed;
    }
}

function updateAIPaddle() {
    const paddleCenter = aiPaddle.y + aiPaddle.height / 2;
    const ballY = ball.y;
    
    // AI follows ball with some delay for realistic difficulty
    if (paddleCenter < ballY - 35) {
        aiPaddle.y += aiPaddle.speed;
    } else if (paddleCenter > ballY + 35) {
        aiPaddle.y -= aiPaddle.speed;
    }
    
    // Keep AI paddle within bounds
    if (aiPaddle.y < 0) aiPaddle.y = 0;
    if (aiPaddle.y > game.height - aiPaddle.height) {
        aiPaddle.y = game.height - aiPaddle.height;
    }
}

function updateBall() {
    ball.x += ball.speedX;
    ball.y += ball.speedY;
    
    // Ball collision with top and bottom walls
    if (ball.y - ball.radius <= 0 || ball.y + ball.radius >= game.height) {
        ball.speedY = -ball.speedY;
        ball.y = ball.y - ball.radius <= 0 ? ball.radius : game.height - ball.radius;
    }
    
    // Ball collision with player paddle
    if (ball.x - ball.radius <= playerPaddle.x + playerPaddle.width &&
        ball.x + ball.radius >= playerPaddle.x &&
        ball.y + ball.radius >= playerPaddle.y &&
        ball.y - ball.radius <= playerPaddle.y + playerPaddle.height &&
        ball.speedX < 0) {
        
        // Calculate hit position (relative to paddle center)
        const hitPos = (ball.y - (playerPaddle.y + playerPaddle.height / 2)) / (playerPaddle.height / 2);
        
        ball.speedX = Math.abs(ball.speedX);
        ball.speedY = hitPos * 8; // Angle variation based on hit position
        
        // Increase speed slightly
        if (Math.abs(ball.speedX) < ball.maxSpeed) {
            ball.speedX *= 1.05;
        }
        if (Math.abs(ball.speedY) < ball.maxSpeed) {
            ball.speedY *= 1.05;
        }
    }
    
    // Ball collision with AI paddle
    if (ball.x + ball.radius >= aiPaddle.x &&
        ball.x - ball.radius <= aiPaddle.x + aiPaddle.width &&
        ball.y + ball.radius >= aiPaddle.y &&
        ball.y - ball.radius <= aiPaddle.y + aiPaddle.height &&
        ball.speedX > 0) {
        
        // Calculate hit position (relative to paddle center)
        const hitPos = (ball.y - (aiPaddle.y + aiPaddle.height / 2)) / (aiPaddle.height / 2);
        
        ball.speedX = -Math.abs(ball.speedX);
        ball.speedY = hitPos * 8; // Angle variation based on hit position
        
        // Increase speed slightly
        if (Math.abs(ball.speedX) < ball.maxSpeed) {
            ball.speedX *= 1.05;
        }
        if (Math.abs(ball.speedY) < ball.maxSpeed) {
            ball.speedY *= 1.05;
        }
    }
    
    // Ball goes out of bounds - scoring
    if (ball.x < -ball.radius) {
        // AI scores
        game.aiScore++;
        updateScore();
        resetBall();
        
        if (game.aiScore >= game.winScore) {
            showGameOver('ai');
            return;
        }
    } else if (ball.x > game.width + ball.radius) {
        // Player scores
        game.playerScore++;
        updateScore();
        resetBall();
        
        if (game.playerScore >= game.winScore) {
            showGameOver('player');
            return;
        }
    }
}

// Rendering
function drawRect(x, y, width, height, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, width, height);
}

function drawCircle(x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
}

function drawDashedLine() {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(game.width / 2, 0);
    ctx.lineTo(game.width / 2, game.height);
    ctx.stroke();
    ctx.setLineDash([]);
}

function render() {
    // Clear canvas
    ctx.clearRect(0, 0, game.width, game.height);
    
    // Draw center line
    drawDashedLine();
    
    // Draw paddles
    drawRect(playerPaddle.x, playerPaddle.y, playerPaddle.width, playerPaddle.height, playerPaddle.color);
    drawRect(aiPaddle.x, aiPaddle.y, aiPaddle.width, aiPaddle.height, aiPaddle.color);
    
    // Draw ball
    drawCircle(ball.x, ball.y, ball.radius, ball.color);
}

// Game Loop
function gameLoop(currentTime = 0) {
    if (gameState !== 'playing') return;
    
    const deltaTime = currentTime - lastTime;
    lastTime = currentTime;
    
    // Update game objects
    updatePlayerPaddle();
    updateAIPaddle();
    updateBall();
    
    // Render everything
    render();
    
    // Continue the loop
    animationId = requestAnimationFrame(gameLoop);
}

// Initialize the game
function init() {
    resetGame();
    updateGameStatus();
    render();
}

// Start the game
init();