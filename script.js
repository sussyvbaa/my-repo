const gameBoard = document.getElementById('game-board');
const scoreDisplay = document.getElementById('score');
const gameOverOverlay = document.getElementById('game-over-overlay');
const winningOverlay = document.getElementById('winning-overlay');
const homeMenu = document.getElementById('home-menu');
const gameArea = document.getElementById('game-area');
const playButton = document.getElementById('play-button');
const aiPlayButton = document.getElementById('ai-play-button');
const aiVisualizer = document.getElementById('ai-visualizer');
const nnCanvas = document.getElementById('nn-canvas');
const nnContext = nnCanvas.getContext('2d');


const boardSize = 4;
let board = [];
let score = 0;
let isGameOver = false;
let hasWon = false;
let aiInterval = null; // To store the interval ID for AI moves
let neuralNetwork = null; // To store the neural network instance

// Initialize the game board
function initializeBoard() {
    board = Array(boardSize).fill(null).map(() => Array(boardSize).fill(0));
    score = 0;
    isGameOver = false;
    hasWon = false;
    updateScore();
    renderBoard();
    addNewTile();
    addNewTile(); // Start with two tiles
    gameOverOverlay.classList.remove('visible');
    winningOverlay.classList.remove('visible');
}

// Show the home menu and hide the game area and visualizer
function showMenu() {
    homeMenu.classList.remove('hidden');
    gameArea.classList.add('hidden');
    aiVisualizer.classList.add('hidden');
    if (aiInterval) {
        clearInterval(aiInterval); // Stop AI moves if returning to menu
    }
}

// Show the game area and hide the home menu and visualizer
function showGame() {
    homeMenu.classList.add('hidden');
    gameArea.classList.remove('hidden');
    aiVisualizer.classList.add('hidden');
    initializeBoard(); // Start a new game when showing the game area
}

// Show the game area and visualizer and hide the home menu
function showAIGame() {
    homeMenu.classList.add('hidden');
    gameArea.classList.remove('hidden');
    aiVisualizer.classList.remove('hidden');
    initializeBoard(); // Start a new game for AI
    initializeNeuralNetwork(); // Initialize the neural network
    initializeNeuralNetworkVisualizer(); // Setup visualizer
    aiInterval = setInterval(aiMove, 500); // Start AI moves
}


// Event listeners for menu buttons
playButton.addEventListener('click', showGame);
aiPlayButton.addEventListener('click', showAIGame);

// Initially show the menu
showMenu();

// Basic Neural Network Structure
class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        // Initialize weights and biases with random values
        this.weights_ih = this.createMatrix(this.hiddenNodes, this.inputNodes);
        this.weights_ho = this.createMatrix(this.outputNodes, this.hiddenNodes);
        this.bias_h = this.createMatrix(this.hiddenNodes, 1);
        this.bias_o = this.createMatrix(this.outputNodes, 1);

        this.randomizeMatrix(this.weights_ih);
        this.randomizeMatrix(this.weights_ho);
        this.randomizeMatrix(this.bias_h);
        this.randomizeMatrix(this.bias_o);

        // Learning rate
        this.learning_rate = 0.1; // Example learning rate

        console.log('Neural Network created:', inputNodes, hiddenNodes, outputNodes);
    }

    // Create a matrix with given rows and columns
    createMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = 0;
            }
        }
        return matrix;
    }

    // Randomize matrix values between -1 and 1
    randomizeMatrix(matrix) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    // Matrix multiplication
    multiplyMatrices(matrixA, matrixB) {
        // Check if multiplication is possible
        if (matrixA[0].length !== matrixB.length) {
            console.error("Matrix multiplication not possible!");
            return null;
        }

        const result = this.createMatrix(matrixA.length, matrixB[0].length);

        for (let i = 0; i < result.length; i++) {
            for (let j = 0; j < result[i].length; j++) {
                let sum = 0;
                for (let k = 0; k < matrixA[0].length; k++) {
                    sum += matrixA[i][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // Add matrices
    addMatrices(matrixA, matrixB) {
        if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
            console.error("Matrix addition not possible!");
            return null;
        }
        const result = this.createMatrix(matrixA.length, matrixA[0].length);
        for (let i = 0; i < result.length; i++) {
            for (let j = 0; j < result[i].length; j++) {
                result[i][j] = matrixA[i][j] + matrixB[i][j];
            }
        }
        return result;
    }

    // Apply a function to each element of a matrix
    mapMatrix(matrix, func) {
        const result = this.createMatrix(matrix.length, matrix[0].length);
        for (let i = 0; i < result.length; i++) {
            for (let j = 0; j < result[i].length; j++) {
                result[i][j] = func(matrix[i][j]);
            }
        }
        return result;
    }

    // Transpose a matrix
    transposeMatrix(matrix) {
        const result = this.createMatrix(matrix[0].length, matrix.length);
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    // Scalar multiplication of a matrix
    scalarMultiplyMatrix(matrix, scalar) {
        const result = this.createMatrix(matrix.length, matrix[0].length);
        for (let i = 0; i < result.length; i++) {
            for (let j = 0; j < result[i].length; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }
        return result;
    }


    // Sigmoid activation function
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid function
    dsigmoid(y) {
        return y * (1 - y);
    }


    // Forward propagation
    feedforward(inputsArray) {
        // Convert inputs array to a matrix
        const inputs = this.createMatrix(this.inputNodes, 1);
        for(let i = 0; i < this.inputNodes; i++) {
            inputs[i][0] = inputsArray[i];
        }

        // Calculate hidden layer outputs
        let hidden_inputs = this.multiplyMatrices(this.weights_ih, inputs);
        let hidden_outputs = this.addMatrices(hidden_inputs, this.bias_h);
        hidden_outputs = this.mapMatrix(hidden_outputs, this.sigmoid);

        // Calculate output layer outputs
        let output_inputs = this.multiplyMatrices(this.weights_ho, hidden_outputs);
        let outputs = this.addMatrices(output_inputs, this.bias_o);
        outputs = this.mapMatrix(outputs, this.sigmoid);

        // Convert outputs matrix back to an array
        const outputsArray = [];
        for(let i = 0; i < this.outputNodes; i++) {
            outputsArray.push(outputs[i][0]);
        }
        return outputsArray;
    }

    // Training with backpropagation
    train(inputsArray, targetsArray) {
        // Convert inputs and targets arrays to matrices
        const inputs = this.createMatrix(this.inputNodes, 1);
        for(let i = 0; i < this.inputNodes; i++) {
            inputs[i][0] = inputsArray[i];
        }

        const targets = this.createMatrix(this.outputNodes, 1);
         // Assuming targetsArray is a single value (the correct move index)
         // Need to convert this to a one-hot encoded vector for training
         for(let i = 0; i < this.outputNodes; i++) {
             targets[i][0] = targetsArray[i]; // targetsArray is now a one-hot encoded array
         }


        // --- Forward Pass (same as feedforward) ---
        // Calculate hidden layer outputs
        let hidden_inputs = this.multiplyMatrices(this.weights_ih, inputs);
        let hidden_outputs = this.addMatrices(hidden_inputs, this.bias_h);
        hidden_outputs = this.mapMatrix(hidden_outputs, this.sigmoid);

        // Calculate output layer outputs
        let output_inputs = this.multiplyMatrices(this.weights_ho, hidden_outputs);
        let outputs = this.addMatrices(output_inputs, this.bias_o);
        outputs = this.mapMatrix(outputs, this.sigmoid);

        // --- Backpropagation ---
        // Calculate output layer errors
        let output_errors = this.createMatrix(this.outputNodes, 1);
        for(let i = 0; i < this.outputNodes; i++) {
            output_errors[i][0] = targets[i][0] - outputs[i][0];
        }

        // Calculate output gradients
        let output_gradients = this.mapMatrix(outputs, this.dsigmoid);
        output_gradients = this.hadamard(output_gradients, output_errors); // Element-wise multiplication (Hadamard product)
        output_gradients = this.scalarMultiplyMatrix(output_gradients, this.learning_rate); // Apply learning rate


        // Calculate hidden to output deltas (change in weights)
        let hidden_T = this.transposeMatrix(hidden_outputs);
        let weights_ho_delta = this.multiplyMatrices(output_gradients, hidden_T);

        // Update hidden to output weights and biases
        this.weights_ho = this.addMatrices(this.weights_ho, weights_ho_delta);
        this.bias_o = this.addMatrices(this.bias_o, output_gradients);


        // Calculate hidden layer errors
        let weights_ho_T = this.transposeMatrix(this.weights_ho);
        let hidden_errors = this.multiplyMatrices(weights_ho_T, output_errors);

        // Calculate hidden gradients
        let hidden_gradients = this.mapMatrix(hidden_outputs, this.dsigmoid);
        hidden_gradients = this.hadamard(hidden_gradients, hidden_errors); // Element-wise multiplication
        hidden_gradients = this.scalarMultiplyMatrix(hidden_gradients, this.learning_rate); // Apply learning rate


        // Calculate input to hidden deltas
        let inputs_T = this.transposeMatrix(inputs);
        let weights_ih_delta = this.multiplyMatrices(hidden_gradients, inputs_T);

        // Update input to hidden weights and biases
        this.weights_ih = this.addMatrices(this.weights_ih, weights_ih_delta);
        this.bias_h = this.addMatrices(this.bias_h, hidden_gradients);


        console.log('Training complete.');
    }

    // Helper for element-wise matrix multiplication (Hadamard product)
    hadamard(matrixA, matrixB) {
         if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
             console.error("Hadamard product not possible!");
             return null;
         }
         const result = this.createMatrix(matrixA.length, matrixA[0].length);
         for (let i = 0; i < result.length; i++) {
             for (let j = 0; j < result[i].length; j++) {
                 result[i][j] = matrixA[i][j] * matrixB[i][j];
             }
         }
         return result;
    }
}

// Initialize the neural network (Placeholder)
function initializeNeuralNetwork() {
    const inputNodes = boardSize * boardSize; // 16 input nodes for a 4x4 board
    const hiddenNodes = 64; // Example number of hidden nodes
    const outputNodes = 4; // 4 output nodes for 4 possible moves (Up, Down, Left, Right)
    neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes);
}


// Neural network visualization
function initializeNeuralNetworkVisualizer() {
    console.log('Initializing neural network visualizer...');
    nnCanvas.width = aiVisualizer.clientWidth;
    nnCanvas.height = 300; // Match CSS height
    nnContext.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

    if (!neuralNetwork) {
        console.error("Neural network not initialized for visualization.");
        return;
    }

    const canvasWidth = nnCanvas.width;
    const canvasHeight = nnCanvas.height;
    const nodeRadius = 10;
    const layerGap = 100; // Horizontal space between layers

    // Calculate vertical spacing for nodes
    const inputNodeSpacing = canvasHeight / (neuralNetwork.inputNodes + 1);
    const hiddenNodeSpacing = canvasHeight / (neuralNetwork.hiddenNodes + 1);
    const outputNodeSpacing = canvasHeight / (neuralNetwork.outputNodes + 1);

    // Draw nodes
    nnContext.fillStyle = '#776e65';
    // Input layer
    for (let i = 0; i < neuralNetwork.inputNodes; i++) {
        const x = layerGap;
        const y = (i + 1) * inputNodeSpacing;
        nnContext.beginPath();
        nnContext.arc(x, y, nodeRadius, 0, Math.PI * 2);
        nnContext.fill();
    }
    // Hidden layer
    for (let i = 0; i < neuralNetwork.hiddenNodes; i++) {
        const x = layerGap + 200; // Adjust horizontal position
        const y = (i + 1) * hiddenNodeSpacing;
        nnContext.beginPath();
        nnContext.arc(x, y, nodeRadius, 0, Math.PI * 2);
        nnContext.fill();
    }
    // Output layer
    for (let i = 0; i < neuralNetwork.outputNodes; i++) {
        const x = layerGap + 400; // Adjust horizontal position
        const y = (i + 1) * outputNodeSpacing;
        nnContext.beginPath();
        nnContext.arc(x, y, nodeRadius, 0, Math.PI * 2);
        nnContext.fill();
    }

    // Draw connections (simplified - just lines between layers)
    nnContext.strokeStyle = '#bbada0';
    nnContext.lineWidth = 1;
    // Input to hidden connections
    for (let i = 0; i < neuralNetwork.inputNodes; i++) {
        for (let j = 0; j < neuralNetwork.hiddenNodes; j++) {
            const x1 = layerGap + nodeRadius;
            const y1 = (i + 1) * inputNodeSpacing;
            const x2 = layerGap + 200 - nodeRadius;
            const y2 = (j + 1) * hiddenNodeSpacing;
            nnContext.beginPath();
            nnContext.moveTo(x1, y1);
            nnContext.lineTo(x2, y2);
            nnContext.stroke();
        }
    }
    // Hidden to output connections
    for (let i = 0; i < neuralNetwork.hiddenNodes; i++) {
        for (let j = 0; j < neuralNetwork.outputNodes; j++) {
            const x1 = layerGap + 200 + nodeRadius;
            const y1 = (i + 1) * hiddenNodeSpacing;
            const x2 = layerGap + 400 - nodeRadius;
            const y2 = (j + 1) * outputNodeSpacing;
            nnContext.beginPath();
            nnContext.moveTo(x1, y1);
            nnContext.lineTo(x2, y2);
            nnContext.stroke();
        }
    }
}

function updateNeuralNetworkVisualizer() {
    // TODO: Implement updating the visualizer based on AI activity (e.g., highlighting active nodes/connections)
    console.log('Updating neural network visualizer...');
}


// Function to start AI Play mode
function startAIPlay() {
    showAIGame(); // Show the game board and visualizer
    // TODO: Implement AI logic and visualizer
    // For now, a simple random move AI
    // aiInterval = setInterval(aiMove, 500); // Moved to showAIGame
}

// AI makes a move using the neural network
function aiMove() {
    if (isGameOver || hasWon) {
        if (aiInterval) {
            clearInterval(aiInterval); // Stop AI moves if game is over or won
        }
        return;
    }

    // Prepare board state as input for the neural network
    const inputs = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            inputs.push(board[r][c]); // Use tile values as input
        }
    }

    // Get move probabilities from the neural network
    const moveProbabilities = neuralNetwork.feedforward(inputs);

    // Select a move based on the probabilities (e.g., choose the move with the highest probability)
    const possibleMoves = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
    let bestMoveIndex = 0;
    for (let i = 1; i < moveProbabilities.length; i++) {
        if (moveProbabilities[i] > moveProbabilities[bestMoveIndex]) {
            bestMoveIndex = i;
        }
    }
    const chosenMove = possibleMoves[bestMoveIndex];

    // Execute the chosen move
    // Store the board state before the move for reward calculation
    const boardBeforeMove = JSON.parse(JSON.stringify(board));
    const scoreBeforeMove = score;

    handleKeyPress({ key: chosenMove });

    // Calculate the reward based on the move's outcome
    const reward = calculateReward(boardBeforeMove, board, score - scoreBeforeMove);

    // Prepare target for training (using the reward)
    // A simple approach is to reinforce the chosen move based on the reward.
    // A more sophisticated approach would use Q-learning or other reinforcement learning techniques.
    const targets = [0, 0, 0, 0];
    targets[bestMoveIndex] = reward > 0 ? 1 : 0; // Simple reinforcement: reinforce good moves


    // Train the neural network
    const trainingInputs = []; // Use a different variable name
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            trainingInputs.push(boardBeforeMove[r][c]); // Use board state BEFORE the move as input for training
        }
    }
    neuralNetwork.train(trainingInputs, targets);


    // Update the visualizer (placeholder)
    updateNeuralNetworkVisualizer();
}

// Calculate a reward based on the move's outcome
function calculateReward(boardBefore, boardAfter, scoreChange) {
    let reward = scoreChange; // Reward for points gained

    // Penalize moves that don't change the board
    if (JSON.stringify(boardBefore) === JSON.stringify(boardAfter)) {
        reward -= 10; // Example penalty
    }

    // TODO: Add more sophisticated reward components (e.g., reward for creating empty cells,
    // penalize for creating undesirable board configurations).

    return reward;
}

// Helper function to simulate a move on a given board (without modifying the actual game board)
function simulateMove(simBoard, move) {
    let boardBefore = JSON.stringify(simBoard);
    let simulatedScore = 0; // Local score for simulation

    function slideAndMergeSim(arr) {
        let filteredArr = arr.filter(num => num !== 0);
        let newArr = Array(boardSize).fill(0);

        for (let i = 0; i < filteredArr.length - 1; i++) {
            if (filteredArr[i] === filteredArr[i + 1]) {
                filteredArr[i] *= 2;
                simulatedScore += filteredArr[i];
                filteredArr[i + 1] = 0;
            }
        }
        filteredArr = filteredArr.filter(num => num !== 0);
        for (let i = 0; i < filteredArr.length; i++) {
            newArr[i] = filteredArr[i];
        }
        return newArr;
    }

    switch (move) {
        case 'ArrowUp':
            for (let c = 0; c < boardSize; c++) {
                let column = [];
                for (let r = 0; r < boardSize; r++) {
                    column.push(simBoard[r][c]);
                }
                column = slideAndMergeSim(column);
                for (let r = 0; r < boardSize; r++) {
                    simBoard[r][c] = column[r];
                }
            }
            break;
        case 'ArrowDown':
            for (let c = 0; c < boardSize; c++) {
                let column = [];
                for (let r = 0; r < boardSize; r++) {
                    column.push(simBoard[r][c]);
                }
                column.reverse();
                column = slideAndMergeSim(column);
                column.reverse();
                for (let r = 0; r < boardSize; r++) {
                    simBoard[r][c] = column[r];
                }
            }
            break;
        case 'ArrowLeft':
            for (let r = 0; r < boardSize; r++) {
                simBoard[r] = slideAndMergeSim(simBoard[r]);
            }
            break;
        case 'ArrowRight':
            for (let r = 0; r < boardSize; r++) {
                simBoard[r].reverse();
                simBoard[r] = slideAndMergeSim(simBoard[r]);
                simBoard[r].reverse();
            }
            break;
    }

    // Add simulated score to the actual score for evaluation
    score += simulatedScore;

    return boardBefore !== JSON.stringify(simBoard); // Return true if board changed
}


// Helper function to count empty cells on a board
function countEmptyCells(board) {
    let count = 0;
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Helper function to get the highest tile value on a board
function getHighestTile(board) {
    let highest = 0;
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] > highest) {
                highest = board[r][c];
            }
        }
    }
    return highest;
}

// Render the board in the HTML
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < boardSize - 1 && board[r + 1][c] === currentValue) {
                    return true;
                }
            }
        }
    }
    return false;
}


// Move and merge functions
function moveUp() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column = slideAndMerge(column);
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveDown() {
    for (let c = 0; c < boardSize; c++) {
        let column = [];
        for (let r = 0; r < boardSize; r++) {
            column.push(board[r][c]);
        }
        column.reverse();
        column = slideAndMerge(column);
        column.reverse();
        for (let r = 0; r < boardSize; r++) {
            board[r][c] = column[r];
        }
    }
}

function moveLeft() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row = slideAndMerge(row);
        board[r] = row;
    }
}

function moveRight() {
    for (let r = 0; r < boardSize; r++) {
        let row = board[r];
        row.reverse();
        row = slideAndMerge(row);
        row.reverse();
        board[r] = row;
    }
}

// Helper function to remove zeros from an array
function filterZero(arr) {
    return arr.filter(num => num !== 0);
}

// Helper function to slide and merge tiles in a single row or column
function slideAndMerge(arr) {
    let filteredArr = filterZero(arr);
    let newArr = Array(boardSize).fill(0);

    // Merge
    let mergedIndices = [];
    for (let i = 0; i < filteredArr.length - 1; i++) {
        if (filteredArr[i] === filteredArr[i + 1]) {
            filteredArr[i] *= 2;
            score += filteredArr[i];
            filteredArr[i + 1] = 0;
            mergedIndices.push(i); // Store index of merged tile
        }
    }

    // Slide again after merging
    filteredArr = filterZero(filteredArr);

    // Place merged tiles in the new array
    for (let i = 0; i < filteredArr.length; i++) {
        newArr[i] = filteredArr[i];
    }

    // Add 'merged' class to merged tiles for animation
    // This part needs to be handled after rendering the board
    // The current structure makes it difficult to directly target the new tile element here.
    // A better approach would be to handle animations in the renderBoard function
    // based on changes in the board state.

    return newArr;
}

// Modify renderBoard to handle animations based on board changes
function renderBoard() {
    gameBoard.innerHTML = ''; // Clear the board before rendering
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const tileValue = board[r][c];
            const tile = document.createElement('div');
            tile.classList.add('tile');
            if (tileValue > 0) {
                tile.textContent = tileValue;
                tile.classList.add('tile-' + tileValue); // Add class for styling
                // Position tiles for animation
                tile.style.top = `${r * (100 + 15) + 15}px`;
                tile.style.left = `${c * (100 + 15) + 15}px`;
            } else {
                // Create an empty cell placeholder for consistent grid
                 tile.style.backgroundColor = '#cdc1b4'; // Background for empty cells
                 tile.style.top = `${r * (100 + 15) + 15}px`;
                 tile.style.left = `${c * (100 + 15) + 15}px`;
            }
            gameBoard.appendChild(tile);
        }
    }
}

// Get color for tile based on its value (no longer needed with CSS classes)
// function getTileColor(value) { ... }

// Get text color for tile based on its value (no longer needed with CSS classes)
// function getTextColor(value) { ... }

// Add a new tile (either 2 or 4) to a random empty cell
function addNewTile() {
    let emptyCells = [];
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                emptyCells.push({ r, c });
            }
        }
    }

    if (emptyCells.length > 0) {
        const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        board[randomCell.r][randomCell.c] = Math.random() < 0.9 ? 2 : 4;
        renderBoard();
    }
}

// Update the score display
function updateScore() {
    scoreDisplay.textContent = score;
}

// Handle keyboard input
document.addEventListener('keydown', handleKeyPress);

function handleKeyPress(event) {
    if (isGameOver || hasWon) {
        return; // Don't process input if game is over or won
    }

    switch (event.key) {
        case 'ArrowUp':
            moveUp();
            break;
        case 'ArrowDown':
            moveDown();
            break;
        case 'ArrowLeft':
            moveLeft();
            break;
        case 'ArrowRight':
            moveRight();
            break;
    }
    renderBoard();
    addNewTile();
    updateScore();
    checkGameStatus();
}

// Check game status (win or game over)
function checkGameStatus() {
    if (!hasWon) {
        for (let r = 0; r < boardSize; r++) {
            for (let c = 0; c < boardSize; c++) {
                if (board[r][c] === 2048) {
                    hasWon = true;
                    winningOverlay.classList.add('visible');
                    return;
                }
            }
        }
    }


    if (hasEmptyCells() || hasPossibleMerges()) {
        return; // Game is not over
    } else {
        isGameOver = true;
        gameOverOverlay.classList.add('visible');
    }
}

// Check if there are any empty cells
function hasEmptyCells() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            if (board[r][c] === 0) {
                return true;
            }
        }
    }
    return false;
}

// Check if there are any possible merges
function hasPossibleMerges() {
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const currentValue = board[r][c];
            if (currentValue !== 0) {
                // Check right
                if (c < boardSize - 1 && board[r][c + 1] === currentValue) {
                    return true;
                }
                // Check down
                if (r < board
