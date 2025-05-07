import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

const trainModel = (): void => {
    const baseDir = process.cwd();
    const pythonScriptPath = path.resolve(baseDir, 'sentimentAnalysis.py');
    const datasetPath = path.resolve(baseDir, 'data', 'news_dataset.csv');

    // Verify files exist before spawning
    if (!fs.existsSync(pythonScriptPath)) {
        console.error(`Python script not found: ${pythonScriptPath}`);
        return;
    }

    if (!fs.existsSync(datasetPath)) {
        console.error(`Dataset not found: ${datasetPath}`);
        return;
    }

    const pythonProcess = spawn('python', [
        pythonScriptPath,
        datasetPath,
        '--model_path', path.resolve(baseDir, 'sentiment_model.pkl'),
        '--vectorizer_path', path.resolve(baseDir, 'vectorizer.pkl'),
    ]);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Model Training Output: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Model Training Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            console.log('Sentiment Analysis Model Training Completed Successfully');
        } else {
            console.error(`Model Training Failed with Exit Code: ${code}`);
        }
    });
};

// Execute the training
trainModel();