# Ping Pong Game

A classic HTML5 Canvas-based Ping Pong (Pong) game with player vs AI gameplay, realistic ball physics, and smooth 60 FPS performance.

## Features

### Core Gameplay
- **Player vs AI**: Single-player mode with intelligent AI opponent
- **Realistic Ball Physics**: Ball bounces with angle variations based on paddle hit position
- **Dynamic Speed**: Ball speed increases gradually during gameplay for added challenge
- **Score Tracking**: First player to reach 10 points wins
- **Smooth Animation**: 60 FPS gameplay using requestAnimationFrame

### Game States
- **Menu**: Initial state with instructions
- **Playing**: Active gameplay
- **Paused**: Game can be paused and resumed
- **Game Over**: Win/lose screen with restart option

### Visual Design
- **Modern UI**: Clean, minimalist design with glassmorphism effects
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Animated Elements**: Smooth transitions and hover effects
- **Real-time Score Display**: Live score updates during gameplay

## How to Play

### Setup
1. Open `index.html` in any modern web browser
2. The game will load automatically

### Controls
- **W Key**: Move player paddle up
- **S Key**: Move player paddle down
- **Space Bar**: Start game / Pause / Resume
- **Enter Key**: Restart game (during game over)

### Gameplay
1. Press **Space** to start the game
2. Use **W** and **S** keys to control your paddle (green, left side)
3. The AI controls the red paddle on the right side
4. Keep the ball in play by hitting it with your paddle
5. Score points when the ball passes your opponent's paddle
6. First to reach **10 points** wins!

### Game Mechanics
- **Ball Angle**: The ball's bounce angle depends on where it hits the paddle
  - Hit near the center: Straight bounce
  - Hit near the edges: Angled bounce
- **Speed Increase**: Ball speed increases slightly with each paddle hit
- **AI Difficulty**: The AI has a slight delay in its movement for fair gameplay

## Technical Implementation

### Technologies Used
- **HTML5 Canvas**: For game rendering and graphics
- **JavaScript ES6**: Game logic and physics
- **CSS3**: Styling with modern features like backdrop-filter
- **Responsive Design**: Mobile-friendly layout

### Game Architecture
- **Game Loop**: Uses requestAnimationFrame for smooth 60 FPS performance
- **Collision Detection**: Precise rectangular and circular collision detection
- **State Management**: Clean separation of game states (menu, playing, paused, game over)
- **Input Handling**: Responsive keyboard input with proper event management

### Performance Features
- **Optimized Rendering**: Only redraws when necessary
- **Efficient Collision Detection**: Minimal computational overhead
- **Smooth Animation**: Consistent frame timing
- **Memory Management**: Proper cleanup of animation frames

## File Structure

```
├── index.html          # Main HTML structure
├── style.css           # Game styling and responsive design
├── game.js             # Core game logic and physics
└── README.md           # This documentation file
```

## Browser Compatibility

- **Chrome**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support
- **Mobile Browsers**: Responsive design supports touch devices

## Customization

### Modifying Game Settings
You can easily customize the game by modifying variables in `game.js`:

```javascript
// Game difficulty
const game = {
    winScore: 10        // Change win condition
};

// Ball settings
const ball = {
    maxSpeed: 12        // Maximum ball speed
};

// Paddle settings
const playerPaddle = {
    speed: 7            // Player paddle speed
};

const aiPaddle = {
    speed: 5            // AI paddle speed (lower = easier)
};
```

### Visual Customization
Modify colors and styling in `style.css`:

```css
/* Change game colors */
:root {
    --player-color: #4CAF50;    /* Player paddle color */
    --ai-color: #f44336;        /* AI paddle color */
    --ball-color: #ffffff;      /* Ball color */
}
```

## Development Notes

### Adding Features
The game is designed with modularity in mind. You can easily add:
- Sound effects (add audio elements and play them on events)
- Power-ups (modify ball or paddle properties temporarily)
- Multiple difficulty levels (adjust AI speed and reaction time)
- Two-player mode (replace AI logic with second player input)

### Known Limitations
- No sound effects (can be added by integrating Web Audio API)
- Single AI difficulty level (easily customizable)
- No online multiplayer (local play only)

## Credits

Created as a complete HTML5 Canvas game implementation featuring:
- Responsive design for all screen sizes
- Smooth 60 FPS gameplay
- Realistic physics simulation
- Modern UI with glassmorphism effects

Enjoy playing! 🏓