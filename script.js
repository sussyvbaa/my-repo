// DOM Elements
const gameBoardElement = document.getElementById('game-board');
const scoreDisplay = document.getElementById('score');
const gameOverOverlay = document.getElementById('game-over-overlay');
const winningOverlay = document.getElementById('winning-overlay');
const homeMenu = document.getElementById('home-menu');
const gameArea = document.getElementById('game-area');
const playButton = document.getElementById('play-button');
const aiPlayButton = document.getElementById('ai-play-button');

// AI specific DOM Elements
const aiControlsVisualizerContainer = document.getElementById('ai-controls-visualizer-container');
const aiGameBoardElement = document.getElementById('ai-game-board');
const aiScoreDisplay = document.getElementById('ai-score');
const aiGameOverOverlay = document.getElementById('ai-game-over-overlay');
const aiWinningOverlay = document.getElementById('ai-winning-overlay');
const aiVisualizerElement = document.getElementById('ai-visualizer');
const nnCanvas = document.getElementById('nn-canvas');
const nnContext = nnCanvas.getContext('2d');
const aiSpeedSlider = document.getElementById('ai-speed-slider');
const aiSpeedValue = document.getElementById('ai-speed-value');


// Overlay Buttons
const restartButtonGameOver = document.getElementById('restart-button-gameover');
const menuButtonGameOver = document.getElementById('menu-button-gameover');
const continueButtonWin = document.getElementById('continue-button-win');
const restartButtonWin = document.getElementById('restart-button-win');
const menuButtonWin = document.getElementById('menu-button-win');

const aiRestartButtonGameOver = document.getElementById('ai-restart-button-gameover');
const aiMenuButtonGameOver = document.getElementById('ai-menu-button-gameover');
const aiContinueButtonWin = document.getElementById('ai-continue-button-win');
const aiRestartButtonWin = document.getElementById('ai-restart-button-win');
const aiMenuButtonWin = document.getElementById('ai-menu-button-win');


// Game constants and state
const BOARD_SIZE = 4;
let board = [];
let score = 0;
let isGameOver = false;
let hasWon = false;
let currentMode = 'manual'; // 'manual' or 'ai'

// AI state
let aiInterval = null;
let neuralNetwork = null;
let lastActivations = { inputs: [], hidden: [], outputs: [] };
const EPSILON_START = 0.9; // Start with more exploration
const EPSILON_END = 0.01;  // Minimum exploration
const EPSILON_DECAY = 0.99; // Rate of decay
let epsilon = EPSILON_START;
let aiMoveDelay = 500; // Initial AI speed

// --- Game Initialization and UI Management ---

function initializeGame(mode = 'manual') {
    currentMode = mode;
    board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0));
    score = 0;
    isGameOver = false;
    hasWon = false; // Reset win state for "Keep Going"

    updateScoreDisplay();
    addNewTile();
    addNewTile();
    renderBoard();

    if (mode === 'manual') {
        gameOverOverlay.classList.remove('visible');
        winningOverlay.classList.remove('visible');
    } else { // AI mode
        aiGameOverOverlay.classList.remove('visible');
        aiWinningOverlay.classList.remove('visible');
        epsilon = EPSILON_START; // Reset exploration rate for new AI game
    }
}

function showMenu() {
    homeMenu.classList.remove('hidden');
    gameArea.classList.add('hidden');
    aiControlsVisualizerContainer.classList.add('hidden');
    if (aiInterval) clearInterval(aiInterval);
}

function showGame(isAI = false) {
    homeMenu.classList.add('hidden');
    if (isAI) {
        gameArea.classList.add('hidden');
        aiControlsVisualizerContainer.classList.remove('hidden');
        initializeGame('ai');
        initializeNeuralNetwork();
        updateNeuralNetworkVisualizer(); // Initial draw
        if (aiInterval) clearInterval(aiInterval);
        aiInterval = setInterval(aiMove, aiMoveDelay);
    } else {
        aiControlsVisualizerContainer.classList.add('hidden');
        gameArea.classList.remove('hidden');
        initializeGame('manual');
    }
}

// --- Neural Network Class ---
class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes; // Can be an array for multiple hidden layers
        this.outputNodes = outputNodes;
        this.learning_rate = 0.05; // Adjusted learning rate

        // For simplicity, one hidden layer. For more layers, this needs to be an array of matrices.
        this.weights_ih = this.createMatrix(this.hiddenNodes, this.inputNodes);
        this.weights_ho = this.createMatrix(this.outputNodes, this.hiddenNodes);
        this.bias_h = this.createMatrix(this.hiddenNodes, 1);
        this.bias_o = this.createMatrix(this.outputNodes, 1);

        this.randomizeMatrix(this.weights_ih);
        this.randomizeMatrix(this.weights_ho);
        this.randomizeMatrix(this.bias_h);
        this.randomizeMatrix(this.bias_o);

        // Store activations for visualization
        this.last_input_activations = new Array(this.inputNodes).fill(0);
        this.last_hidden_activations = new Array(this.hiddenNodes).fill(0);
        this.last_output_activations = new Array(this.outputNodes).fill(0);
    }

    createMatrix(rows, cols) {
        return Array(rows).fill(null).map(() => Array(cols).fill(0));
    }

    randomizeMatrix(matrix) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = Math.random() * 2 - 1; // Values between -1 and 1
            }
        }
    }

    // ReLU activation function - often better for hidden layers
    relu(x) {
        return Math.max(0, x);
    }

    // Derivative of ReLU
    drelu(y) {
        return y > 0 ? 1 : 0;
    }

    // Sigmoid activation function (good for output layer for probabilities)
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid
    dsigmoid(y) {
        return y * (1 - y);
    }


    feedforward(inputsArray) {
        // Convert inputs array to a matrix and store
        const inputs = this.createMatrix(this.inputNodes, 1);
        for (let i = 0; i < this.inputNodes; i++) {
            inputs[i][0] = inputsArray[i];
            this.last_input_activations[i] = inputsArray[i];
        }

        // Hidden Layer Calculation (using ReLU)
        let hidden_inputs_raw = this.multiplyMatrices(this.weights_ih, inputs);
        let hidden_outputs_biased = this.addMatrices(hidden_inputs_raw, this.bias_h);
        let hidden_outputs = this.mapMatrix(hidden_outputs_biased, this.relu); // Using ReLU
        for(let i=0; i < this.hiddenNodes; i++) this.last_hidden_activations[i] = hidden_outputs[i][0];


        // Output Layer Calculation (using Sigmoid for probabilities)
        let output_inputs_raw = this.multiplyMatrices(this.weights_ho, hidden_outputs);
        let outputs_biased = this.addMatrices(output_inputs_raw, this.bias_o);
        let final_outputs = this.mapMatrix(outputs_biased, this.sigmoid);
        for(let i=0; i < this.outputNodes; i++) this.last_output_activations[i] = final_outputs[i][0];

        // Update global lastActivations for the visualizer
        lastActivations = {
            inputs: [...this.last_input_activations],
            hidden: [...this.last_hidden_activations],
            outputs: [...this.last_output_activations]
        };

        return final_outputs.map(row => row[0]); // Convert to simple array
    }

    train(inputsArray, targetsArray) {
        // --- Forward Pass (copied from feedforward, essentially) ---
        const inputs = this.createMatrix(this.inputNodes, 1);
        for (let i = 0; i < this.inputNodes; i++) inputs[i][0] = inputsArray[i];

        let hidden_inputs_raw = this.multiplyMatrices(this.weights_ih, inputs);
        let hidden_outputs_biased = this.addMatrices(hidden_inputs_raw, this.bias_h);
        let hidden_outputs = this.mapMatrix(hidden_outputs_biased, this.relu); // ReLU for hidden

        let output_inputs_raw = this.multiplyMatrices(this.weights_ho, hidden_outputs);
        let outputs_biased = this.addMatrices(output_inputs_raw, this.bias_o);
        let final_outputs = this.mapMatrix(outputs_biased, this.sigmoid); // Sigmoid for output

        // Convert targets array to matrix
        const targets = this.createMatrix(this.outputNodes, 1);
        for (let i = 0; i < this.outputNodes; i++) targets[i][0] = targetsArray[i];

        // --- Backpropagation ---
        // Calculate output errors: (target - actual)
        let output_errors = this.subtractMatrices(targets, final_outputs);

        // Calculate output gradients: error * dsigmoid(output)
        let output_gradients = this.mapMatrix(final_outputs, this.dsigmoid);
        output_gradients = this.hadamardProduct(output_gradients, output_errors);
        output_gradients = this.scalarMultiplyMatrix(output_gradients, this.learning_rate);

        // Calculate hidden-to-output weight deltas
        let hidden_outputs_T = this.transposeMatrix(hidden_outputs);
        let weights_ho_deltas = this.multiplyMatrices(output_gradients, hidden_outputs_T);

        // Adjust weights and biases for output layer
        this.weights_ho = this.addMatrices(this.weights_ho, weights_ho_deltas);
        this.bias_o = this.addMatrices(this.bias_o, output_gradients); // Bias delta is just the gradient

        // Calculate hidden layer errors
        let weights_ho_T = this.transposeMatrix(this.weights_ho);
        let hidden_errors = this.multiplyMatrices(weights_ho_T, output_errors);

        // Calculate hidden gradients: hidden_error * drelu(hidden_output)
        let hidden_gradients = this.mapMatrix(hidden_outputs, this.drelu); // Using dReLU
        hidden_gradients = this.hadamardProduct(hidden_gradients, hidden_errors);
        hidden_gradients = this.scalarMultiplyMatrix(hidden_gradients, this.learning_rate);

        // Calculate input-to-hidden weight deltas
        let inputs_T = this.transposeMatrix(inputs);
        let weights_ih_deltas = this.multiplyMatrices(hidden_gradients, inputs_T);

        // Adjust weights and biases for hidden layer
        this.weights_ih = this.addMatrices(this.weights_ih, weights_ih_deltas);
        this.bias_h = this.addMatrices(this.bias_h, hidden_gradients);
    }

    // Matrix math helper functions (add, subtract, multiply, map, transpose, scalar multiply, hadamard)
    addMatrices(m1, m2) { return m1.map((row, i) => row.map((val, j) => val + m2[i][j])); }
    subtractMatrices(m1, m2) { return m1.map((row, i) => row.map((val, j) => val - m2[i][j])); }
    mapMatrix(matrix, func) { return matrix.map(row => row.map(val => func(val))); }
    transposeMatrix(matrix) {
        return matrix[0].map((col, i) => matrix.map(row => row[i]));
    }
    scalarMultiplyMatrix(matrix, scalar) {
        return matrix.map(row => row.map(val => val * scalar));
    }
    hadamardProduct(m1, m2) { // Element-wise multiplication
        return m1.map((row, i) => row.map((val, j) => val * m2[i][j]));
    }
    multiplyMatrices(m1, m2) { // Standard matrix multiplication
        const result = this.createMatrix(m1.length, m2[0].length);
        for (let i = 0; i < result.length; i++) {
            for (let j = 0; j < result[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < m1[0].length; k++) {
                    sum += m1[i][k] * m2[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
}

function initializeNeuralNetwork() {
    const inputNodeCount = BOARD_SIZE * BOARD_SIZE;
    const hiddenNodeCount = 32; // Tunable: More nodes can learn complex patterns but risk overfitting/slower training
    const outputNodeCount = 4;  // Up, Down, Left, Right
    neuralNetwork = new NeuralNetwork(inputNodeCount, hiddenNodeCount, outputNodeCount);
}

// --- AI Logic ---
function normalizeBoardInputs(currentBoard) {
    const inputs = [];
    let maxVal = 0;
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (currentBoard[r][c] > maxVal) maxVal = currentBoard[r][c];
        }
    }
    // Normalize by log2(value + 1) / log2(maxVal + 1) or just scale to 0-1 range
    // Using log helps manage the large differences in tile values. Add 1 to avoid log(0).
    const logMax = maxVal > 0 ? Math.log2(maxVal +1) : 1; // Avoid division by zero if board is empty or logMax is 0

    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const val = currentBoard[r][c];
            inputs.push(val > 0 ? Math.log2(val + 1) / logMax : 0);
        }
    }
    return inputs;
}


function aiMove() {
    if (isGameOver || (hasWon && currentMode === 'ai' && !aiWinningOverlay.classList.contains('visible'))) { // Stop if game over or AI won and overlay is up
        if (aiInterval) clearInterval(aiInterval);
        checkGameStatus(); // Ensure overlay is shown
        return;
    }

    const boardBeforeMove = JSON.parse(JSON.stringify(board)); // Deep copy
    const scoreBeforeMove = score;

    const normalizedInputs = normalizeBoardInputs(board);
    const moveProbabilities = neuralNetwork.feedforward(normalizedInputs);

    let chosenMoveIndex;
    if (Math.random() < epsilon) {
        // Explore: choose a random move
        chosenMoveIndex = Math.floor(Math.random() * 4);
    } else {
        // Exploit: choose the best move from NN output
        chosenMoveIndex = moveProbabilities.indexOf(Math.max(...moveProbabilities));
    }

    // Decay epsilon
    if (epsilon > EPSILON_END) {
        epsilon *= EPSILON_DECAY;
    }


    const possibleMoves = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
    const chosenMoveKey = possibleMoves[chosenMoveIndex];

    // Simulate the move to see if it's valid (changes the board)
    let boardChanged = makeMove(chosenMoveKey, true); // True for simulation

    if (boardChanged) {
        makeMove(chosenMoveKey); // Execute the move on the actual board
        addNewTile();
    }
    renderBoard();
    updateScoreDisplay();
    updateNeuralNetworkVisualizer(); // Update visualizer after NN feedforward and move

    const reward = calculateReward(boardBeforeMove, board, score - scoreBeforeMove, boardChanged);

    // Prepare targets for training (one-hot encoding for the chosen move, scaled by reward)
    const targets = [0, 0, 0, 0];
    if (boardChanged) { // Only train on valid moves that changed the board
        targets[chosenMoveIndex] = reward > 0 ? Math.min(1, reward / 10) : 0; // Scale reward, cap at 1
    }


    neuralNetwork.train(normalizedInputs, targets); // Train with normalized inputs from *before* the move

    checkGameStatus(); // Check game status after AI move
}

function calculateReward(boardBefore, boardAfter, scoreChange, boardChanged) {
    let reward = 0;

    // 1. Score Change (Primary Reward)
    reward += scoreChange * 0.5; // Weighted score change

    // 2. Board Changed
    if (!boardChanged) {
        return -20; // Heavy penalty for invalid moves (that don't change the board)
    }

    // 3. Number of Empty Cells
    const emptyCellsBefore = countEmptyCells(boardBefore);
    const emptyCellsAfter = countEmptyCells(boardAfter);
    if (emptyCellsAfter > emptyCellsBefore) {
        reward += (emptyCellsAfter - emptyCellsBefore) * 2; // Reward for creating empty space
    } else if (emptyCellsAfter < emptyCellsBefore && scoreChange <=0){
        reward -= (emptyCellsBefore - emptyCellsAfter) * 1; // Penalty for losing empty space without score
    }


    // 4. Monotonicity (tiles generally increasing/decreasing along rows/cols)
    // This encourages smoother boards.
    // Simplified: check if highest tile is in a corner.
    const highestTileAfter = getHighestTileValue(boardAfter);
    const corners = [
        boardAfter[0][0], boardAfter[0][BOARD_SIZE - 1],
        boardAfter[BOARD_SIZE - 1][0], boardAfter[BOARD_SIZE - 1][BOARD_SIZE - 1]
    ];
    if (corners.includes(highestTileAfter) && highestTileAfter > 32) { // Only for significant tiles
        reward += 5;
    }

    // 5. Merging high-value tiles
    if (scoreChange > 0) {
        const maxTileBefore = getHighestTileValue(boardBefore);
        const maxTileAfter = getHighestTileValue(boardAfter);
        if (maxTileAfter > maxTileBefore && maxTileAfter >= 64) { // Merged to create a new, larger max tile
            reward += Math.log2(maxTileAfter) * 2; // Logarithmic reward for higher merges
        }
    }

    // 6. Penalty for game over state
    if (isBoardFull(boardAfter) && !canAnyMoveChangeBoard(boardAfter)) {
        reward -= 50; // Significant penalty if the move leads to game over
    }


    return Math.max(-20, Math.min(reward, 20)); // Clamp reward to a reasonable range
}


// --- Neural Network Visualizer ---
function updateNeuralNetworkVisualizer() {
    if (!neuralNetwork || !aiVisualizerElement.classList.contains('hidden')) { // Only draw if NN exists and visualizer is visible
        nnCanvas.width = nnCanvas.clientWidth; // Responsive width
        nnCanvas.height = 300; // Fixed height
        nnContext.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

        const { inputs, hidden, outputs } = lastActivations;
        const inputNodes = neuralNetwork.inputNodes;
        const hiddenNodes = neuralNetwork.hiddenNodes;
        const outputNodes = neuralNetwork.outputNodes;

        const nodeRadius = Math.min(10, nnCanvas.height / (Math.max(inputNodes, hiddenNodes, outputNodes) * 2.5)); // Dynamic radius
        const layerGap = nnCanvas.width / 4;
        const inputX = layerGap * 0.8;
        const hiddenX = layerGap * 2;
        const outputX = layerGap * 3.2;

        // Function to get color based on activation (0 to 1)
        function getActivationColor(activation) {
            const intensity = Math.max(0, Math.min(1, activation)); // Clamp activation
            // Interpolate between a base color (e.g., light gray) and an active color (e.g., game's orange)
            // Base: #cdc1b4 (empty tile), Active: #f2b179 (tile 8)
            const r = Math.round(205 + (242 - 205) * intensity); // CDC1B4 -> F2B179
            const g = Math.round(193 + (177 - 193) * intensity);
            const b = Math.round(180 + (121 - 180) * intensity);
            return `rgb(${r},${g},${b})`;
        }

        // Function to get color/width for weights
        function getWeightStyle(weight) {
            const alpha = Math.min(1, Math.abs(weight) * 2); // Scale opacity by weight magnitude
            const color = weight > 0 ? `rgba(70, 130, 180, ${alpha})` : `rgba(255, 99, 71, ${alpha})`; // Blue for positive, Red for negative
            const lineWidth = Math.min(3, Math.max(0.5, Math.abs(weight))); // Scale line width
            return { color, lineWidth };
        }

        // Draw connections: Hidden to Output
        for (let i = 0; i < hiddenNodes; i++) {
            for (let j = 0; j < outputNodes; j++) {
                const y1 = (nnCanvas.height / (hiddenNodes + 1)) * (i + 1);
                const y2 = (nnCanvas.height / (outputNodes + 1)) * (j + 1);
                const weight = neuralNetwork.weights_ho[j][i];
                const style = getWeightStyle(weight);
                nnContext.beginPath();
                nnContext.moveTo(hiddenX, y1);
                nnContext.lineTo(outputX, y2);
                nnContext.strokeStyle = style.color;
                nnContext.lineWidth = style.lineWidth;
                nnContext.stroke();
            }
        }

        // Draw connections: Input to Hidden
        for (let i = 0; i < inputNodes; i++) {
            for (let j = 0; j < hiddenNodes; j++) {
                const y1 = (nnCanvas.height / (inputNodes + 1)) * (i + 1);
                const y2 = (nnCanvas.height / (hiddenNodes + 1)) * (j + 1);
                const weight = neuralNetwork.weights_ih[j][i];
                const style = getWeightStyle(weight);
                nnContext.beginPath();
                nnContext.moveTo(inputX, y1);
                nnContext.lineTo(hiddenX, y2);
                nnContext.strokeStyle = style.color;
                nnContext.lineWidth = style.lineWidth;
                nnContext.stroke();
            }
        }


        // Draw Input Nodes
        for (let i = 0; i < inputNodes; i++) {
            const y = (nnCanvas.height / (inputNodes + 1)) * (i + 1);
            nnContext.beginPath();
            nnContext.arc(inputX, y, nodeRadius, 0, Math.PI * 2);
            nnContext.fillStyle = getActivationColor(inputs[i] || 0);
            nnContext.fill();
            nnContext.strokeStyle = '#776e65';
            nnContext.lineWidth = 1;
            nnContext.stroke();
        }

        // Draw Hidden Nodes
        for (let i = 0; i < hiddenNodes; i++) {
            const y = (nnCanvas.height / (hiddenNodes + 1)) * (i + 1);
            nnContext.beginPath();
            nnContext.arc(hiddenX, y, nodeRadius, 0, Math.PI * 2);
            nnContext.fillStyle = getActivationColor(hidden[i] || 0);
            nnContext.fill();
            nnContext.strokeStyle = '#776e65';
            nnContext.stroke();
        }

        // Draw Output Nodes
        const moveLabels = ['U', 'D', 'L', 'R'];
        for (let i = 0; i < outputNodes; i++) {
            const y = (nnCanvas.height / (outputNodes + 1)) * (i + 1);
            nnContext.beginPath();
            nnContext.arc(outputX, y, nodeRadius * 1.2, 0, Math.PI * 2); // Slightly larger output nodes
            nnContext.fillStyle = getActivationColor(outputs[i] || 0);
            nnContext.fill();
            nnContext.strokeStyle = '#5a5047'; // Darker stroke for output
            nnContext.lineWidth = 1.5;
            nnContext.stroke();
            // Draw move labels
            nnContext.fillStyle = '#fff';
            nnContext.font = `${nodeRadius * 0.8}px Clear Sans`;
            nnContext.textAlign = 'center';
            nnContext.textBaseline = 'middle';
            nnContext.fillText(moveLabels[i], outputX, y);
        }
    }
}


// --- Game Board Logic (Shared by Manual and AI) ---
function renderBoard() {
    const targetBoardElement = currentMode === 'ai' ? aiGameBoardElement : gameBoardElement;
    targetBoardElement.innerHTML = ''; // Clear previous tiles

    const boardContainerWidth = targetBoardElement.clientWidth; // Get current width of the container
    const tileSize = (boardContainerWidth - (15 * (BOARD_SIZE + 1))) / BOARD_SIZE; // Calculate tile size based on container width and gaps
    const gapSize = 15;

    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const tileValue = board[r][c];
            const tileDiv = document.createElement('div');
            tileDiv.classList.add('tile');

            // Calculate position
            tileDiv.style.width = `${tileSize}px`;
            tileDiv.style.height = `${tileSize}px`;
            tileDiv.style.top = `${r * (tileSize + gapSize) + gapSize}px`;
            tileDiv.style.left = `${c * (tileSize + gapSize) + gapSize}px`;


            if (tileValue > 0) {
                tileDiv.textContent = tileValue;
                tileDiv.classList.add(`tile-${tileValue > 2048 ? 'super' : tileValue}`);
                // Adjust font size dynamically based on tile size (optional, CSS already handles some responsiveness)
                const dynamicFontSize = tileSize * 0.4; // Adjust multiplier as needed
                if (tileValue >= 1000) tileDiv.style.fontSize = `${dynamicFontSize * 0.7}px`;
                else if (tileValue >= 100) tileDiv.style.fontSize = `${dynamicFontSize * 0.8}px`;
                else tileDiv.style.fontSize = `${dynamicFontSize}px`;

            } else {
                tileDiv.classList.add('tile-empty'); // For styling empty cells if needed
            }
            targetBoardElement.appendChild(tileDiv);
        }
    }
}


function addNewTile(specificBoard = board) {
    let emptyCells = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (specificBoard[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }
    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        specificBoard[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;

        // Add animation class to the new tile after rendering
        // This requires finding the specific tile div, which is easier if renderBoard returns tile elements or uses IDs
        // For simplicity, a general "new-tile" class can be added and removed, or handled by CSS transitions on appearance.
        // The current renderBoard clears and redraws, so direct animation on a specific new tile is tricky without more complex DOM handling.
    }
}

function updateScoreDisplay() {
    if (currentMode === 'ai') {
        aiScoreDisplay.textContent = score;
    } else {
        scoreDisplay.textContent = score;
    }
}

// --- Movement Logic ---
function makeMove(key, simulate = false) {
    let boardToChange = simulate ? JSON.parse(JSON.stringify(board)) : board;
    let originalBoardState = JSON.stringify(boardToChange);
    let moveScore = 0;

    switch (key) {
        case 'ArrowUp': moveScore = moveVertical(boardToChange, true); break;
        case 'ArrowDown': moveScore = moveVertical(boardToChange, false); break;
        case 'ArrowLeft': moveScore = moveHorizontal(boardToChange, true); break;
        case 'ArrowRight': moveScore = moveHorizontal(boardToChange, false); break;
    }

    if (!simulate && moveScore > 0) {
        score += moveScore;
    }
    return originalBoardState !== JSON.stringify(boardToChange); // Returns true if board changed
}

function slideAndMergeLine(line, simulate = false) {
    let newLine = line.filter(val => val !== 0); // Remove zeros
    let lineScore = 0;

    for (let i = 0; i < newLine.length - 1; i++) {
        if (newLine[i] === newLine[i + 1]) {
            newLine[i] *= 2;
            if(!simulate) lineScore += newLine[i]; // Add to score only if not simulating
            newLine.splice(i + 1, 1); // Remove merged tile
        }
    }
    // Pad with zeros to the original length
    while (newLine.length < BOARD_SIZE) {
        newLine.push(0);
    }
    return { newLine, lineScore };
}

function moveVertical(boardRef, up) {
    let totalMoveScore = 0;
    for (let c = 0; c < BOARD_SIZE; c++) {
        let column = [];
        for (let r = 0; r < BOARD_SIZE; r++) {
            column.push(boardRef[r][c]);
        }
        if (!up) column.reverse(); // For down movement, process reversed column then reverse back

        const { newLine, lineScore } = slideAndMergeLine(column, boardRef !== board); // Pass simulate flag
        totalMoveScore += lineScore;

        if (!up) newLine.reverse();

        for (let r = 0; r < BOARD_SIZE; r++) {
            boardRef[r][c] = newLine[r];
        }
    }
    return totalMoveScore;
}

function moveHorizontal(boardRef, left) {
    let totalMoveScore = 0;
    for (let r = 0; r < BOARD_SIZE; r++) {
        let row = [...boardRef[r]]; // Copy row
        if (!left) row.reverse(); // For right movement

        const { newLine, lineScore } = slideAndMergeLine(row, boardRef !== board); // Pass simulate flag
        totalMoveScore += lineScore;

        if (!left) newLine.reverse();
        boardRef[r] = newLine;
    }
    return totalMoveScore;
}


// --- Game Status Checks ---
function checkGameStatus() {
    const targetGameOverOverlay = currentMode === 'ai' ? aiGameOverOverlay : gameOverOverlay;
    const targetWinningOverlay = currentMode === 'ai' ? aiWinningOverlay : winningOverlay;

    if (!hasWon) {
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (board[r][c] === 2048) { // Or higher for continued play: e.g. >= 2048
                    hasWon = true;
                    targetWinningOverlay.classList.add('visible');
                    if (aiInterval) clearInterval(aiInterval); // Stop AI if it wins
                    return; // Exit check early if won
                }
            }
        }
    }

    if (isBoardFull(board) && !canAnyMoveChangeBoard(board)) {
        isGameOver = true;
        targetGameOverOverlay.classList.add('visible');
        if (aiInterval) clearInterval(aiInterval); // Stop AI if game over
    }
}

function isBoardFull(currentBoard) {
    return !currentBoard.flat().includes(0);
}

function canAnyMoveChangeBoard(currentBoard) {
    const testKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
    for (const key of testKeys) {
        let tempBoard = JSON.parse(JSON.stringify(currentBoard)); // Deep copy for simulation
        let originalState = JSON.stringify(tempBoard);
        let moveScore = 0; // Temporary variable for simulated move score

        // Simulate move without affecting actual score or board
        switch (key) {
            case 'ArrowUp': moveScore = moveVertical(tempBoard, true); break;
            case 'ArrowDown': moveScore = moveVertical(tempBoard, false); break;
            case 'ArrowLeft': moveScore = moveHorizontal(tempBoard, true); break;
            case 'ArrowRight': moveScore = moveHorizontal(tempBoard, false); break;
        }
        if (originalState !== JSON.stringify(tempBoard)) {
            return true; // A move is possible
        }
    }
    return false; // No move can change the board
}

function countEmptyCells(targetBoard) {
    return targetBoard.flat().filter(val => val === 0).length;
}

function getHighestTileValue(targetBoard) {
    return Math.max(0, ...targetBoard.flat());
}


// --- Event Listeners ---
// Menu Buttons
playButton.addEventListener('click', () => showGame(false));
aiPlayButton.addEventListener('click', () => showGame(true));

// Keyboard Input
document.addEventListener('keydown', (event) => {
    if (event.key === 'm' || event.key === 'M') {
        showMenu();
        return;
    }

    if (currentMode === 'manual') {
        if (gameOverOverlay.classList.contains('visible') || winningOverlay.classList.contains('visible') && !hasWon) { // Allow continue if won
             return;
        }
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
            event.preventDefault(); // Prevent page scrolling
            if (makeMove(event.key)) { // If move was valid and changed the board
                addNewTile();
            }
            renderBoard();
            updateScoreDisplay();
            checkGameStatus();
        }
    }
});

// Overlay Buttons Listeners
function setupOverlayButtonListeners() {
    // Manual Game Over
    restartButtonGameOver.addEventListener('click', () => initializeGame('manual'));
    menuButtonGameOver.addEventListener('click', showMenu);
    // Manual Win
    continueButtonWin.addEventListener('click', () => {
        winningOverlay.classList.remove('visible');
        // hasWon remains true, game continues
    });
    restartButtonWin.addEventListener('click', () => initializeGame('manual'));
    menuButtonWin.addEventListener('click', showMenu);

    // AI Game Over
    aiRestartButtonGameOver.addEventListener('click', () => showGame(true));
    aiMenuButtonGameOver.addEventListener('click', showMenu);
    // AI Win
    aiContinueButtonWin.addEventListener('click', () => {
        aiWinningOverlay.classList.remove('visible');
        // hasWon remains true, AI continues if logic allows
        if (aiInterval) clearInterval(aiInterval); // Clear old one
        aiInterval = setInterval(aiMove, aiMoveDelay); // Restart AI
    });
    aiRestartButtonWin.addEventListener('click', () => showGame(true));
    aiMenuButtonWin.addEventListener('click', showMenu);
}

// AI Speed Control
aiSpeedSlider.addEventListener('input', (e) => {
    aiMoveDelay = parseInt(e.target.value, 10);
    aiSpeedValue.textContent = `${aiMoveDelay} ms`;
    if (currentMode === 'ai' && aiInterval) {
        clearInterval(aiInterval);
        aiInterval = setInterval(aiMove, aiMoveDelay);
    }
});


// Resize listener for rendering board and visualizer
window.addEventListener('resize', () => {
    if (!gameArea.classList.contains('hidden')) {
        renderBoard();
    }
    if (!aiControlsVisualizerContainer.classList.contains('hidden')) {
        renderBoard(); // For AI board
        updateNeuralNetworkVisualizer();
    }
});

// Initial Setup
showMenu();
setupOverlayButtonListeners();
// Call renderBoard once at the start if a game mode is active by default (not in this setup)
// Call updateNeuralNetworkVisualizer if AI mode is active by default
