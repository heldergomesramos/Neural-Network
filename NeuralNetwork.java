import java.util.HashSet;
import java.util.Scanner;

public class NeuralNetwork {

    public InputLayer inputLayer;
    public HiddenLayer[] hiddenLayers;
    public OutputLayer outputLayer;

    public NeuralNetwork(int iNeurons, int[] hNeurons, int oNeurons, HashSet<String> possibleOutputs, double learningRate) {
        inputLayer = new InputLayer(iNeurons);
        hiddenLayers = new HiddenLayer[hNeurons.length];
        hiddenLayers[0] = new HiddenLayer(hNeurons[0], inputLayer, 1, learningRate);
        for(int i = 1; i < hNeurons.length; i++)
            hiddenLayers[i] = new HiddenLayer(hNeurons[i], hiddenLayers[i-1], i + 1, learningRate);
        outputLayer = new OutputLayer(oNeurons, hiddenLayers[hNeurons.length - 1], possibleOutputs, learningRate);
    }

    /* Adapts the number of input and output neurons to the given inputs */
    public static NeuralNetwork inputNeuralNetwork(HashSet<String> possibleOutputs, int iNeurons, Scanner input) {
        System.out.println("Detected from training samples");
        System.out.println(iNeurons + " input neurons");
        System.out.println(possibleOutputs.size() + " output neurons " + possibleOutputs.toString());
        System.out.println("Insert number of hidden layers");
        int hLayers = input.nextInt();
        int[] hNeurons = new int[hLayers];
        for (int i = 0; i < hNeurons.length; i++) {
            System.out.println("Insert number of neurons for hidden layer " + (i + 1));
            hNeurons[i] = input.nextInt();    
        }
        System.out.println("Insert learning rate");
        double learningRate = input.nextDouble();
        System.out.println("Insert time limit (seconds) in the case the stopping condition is not met");
        Main.timeLimit = input.nextInt() * 1000;
        return new NeuralNetwork(iNeurons, hNeurons, possibleOutputs.size(), possibleOutputs, learningRate);
    }
    
    /**
     * Debug function with an opcode for what part of the neural network to show. 0 means not show
     * @param opcode 1XX - show neurons; X1X - show biases; XX1 - show weights
     */
    public void debug(char[] opcode) {
        System.out.println(inputLayer.toString(opcode));
        for (int i = 0; i < hiddenLayers.length; ++i)
            System.out.println(hiddenLayers[i].toString(opcode));
        System.out.println(outputLayer.toString(opcode));
    }

    /**
     * Uses feed forward followed by back propagation for a given training example
     * @param inputs array of values given as inputs
     */
    public int trainNetwork(double[] inputs, String expectedVal) {
        feedForward(inputs);
        int correctIndex = outputLayer.getCorrectIndex(expectedVal);
        outputLayer.backPropagation(correctIndex);
        return outputLayer.getBestIndex() == correctIndex ? 1 : 0;
    }

    public String test(double[] testData) {
        feedForward(testData);
        return outputLayer.getResult();
    }

    public void feedForward(double[] inputs) {
        inputLayer.setInputValues(inputs);
        for (int i = 0; i < hiddenLayers.length; ++i)
            hiddenLayers[i].setUpNeurons();
        outputLayer.setUpNeurons();
    }
}