import java.text.DecimalFormat;
import java.util.HashSet;
import java.util.Scanner;

/**
 * Class representing a layer, from which InputLayer HiddenLayer are extended. Contains most variables and functions related to layers
 */
public abstract class Layer {
    protected DecimalFormat dfh = new DecimalFormat("0.000000");
    protected int numberOfNeurons;
    protected double[] neurons; /* Value for each neuron in the layer */

    public Layer(int numberOfNeurons) {
        this.numberOfNeurons = numberOfNeurons;
        neurons = new double[numberOfNeurons];
    }

    public abstract String toString(char[] opcode);
    public abstract void backPropagation(double[] hErrors);
}

/**
 * Class responsible for the middle layers of the neural network, from which the output layer extends itself
 */
class HiddenLayer extends Layer {
    protected Layer prevLayer; /* Represents the previous layer in the neural network, used to make the weights graph */
    protected double[] biases; /* Biases for each neuron in the layer */
    protected double[][] weights; /* Weights for each connection between the previous Layer i and the current Layer j as in weights[i][j] */
    protected int index;
    protected double learningRate;

    /*
     * Default constructor for random weight and bias generation
     */
    public HiddenLayer(int numberOfNeurons, Layer prevLayer, int i, double learningRate) {
        super(numberOfNeurons);
        this.prevLayer = prevLayer;
        weights = new double[prevLayer.numberOfNeurons][numberOfNeurons];
        biases = new double[numberOfNeurons];
        generateBiases();
        generateWeights();
        this.index = i;
        this.learningRate = learningRate;
    }

    /*
     * Constructor for specific input weights and biases
     */
    public HiddenLayer(int numberOfNeurons, Layer prevLayer, int i, double[][] weights, double[] biases) {
        super(numberOfNeurons);
        this.prevLayer = prevLayer;
        this.weights = weights;
        this.biases = biases;
        this.index = i;
    }  

    public void readWeights(Scanner input) {
        System.out.println("Insert weights for:");
        for(int j = 0; j < prevLayer.numberOfNeurons; j++)
            for (int i = 0; i < numberOfNeurons; i++) {
                System.out.print("[" + j + "][" + i + "] = ");
                weights[j][i] = input.nextDouble();
            }
    }

    public void readBiases(Scanner input) {
        System.out.println("Insert biases for:");
        for (int i = 0; i < numberOfNeurons; i++) {
            System.out.print("[" + i + "] = ");
            biases[i] = input.nextDouble();
        }
    }

    /**
     * Generate the biases for the layer at hand using a random formula and giving them a value between -1 and 1
     */
    public void generateBiases() {
        for (int i = 0; i < numberOfNeurons; ++i)
            biases[i] = 2 * Math.random() - 1;
    }

    /**
     * Generate the weights between the previous layer and the one at hand using a random formula and giving them a value between -1 and 1
     */
    public void generateWeights() {
        generateBiases();
        for (int i = 0; i < prevLayer.numberOfNeurons; ++i)
            for (int j = 0; j < numberOfNeurons; ++j)
                weights[i][j] = 2 * Math.random() - 1;
    }

    /**
     * Generates the neurons of each layer using a specific formula using weights and activations from the previous layer, as well as biases of the current one
     */
    public void setUpNeurons() {
        for (int i = 0; i < numberOfNeurons; ++i) {
            /* neurons[i] = sigmoid((weights[...][i] * prevNeurons[...]) + biases[i]); */
            for (int j = 0; j < prevLayer.numberOfNeurons; ++j) 
                neurons[i] += weights[j][i] * prevLayer.neurons[j];
            neurons[i] += biases[i];
            neurons[i] = sigmoid(neurons[i]);
        }
    }

    public static double sigmoid(double x) {
        return 1.0/(1+Math.exp(-x));
    }

    public void backPropagation(double[] hErrors) {
        /* Calculate errors to propagate to the hidden layers (δH) */
        double[] prevErrors = new double[prevLayer.numberOfNeurons];
        /* j is the previous layer neuron index */
        for(int j = 0; j < prevLayer.numberOfNeurons; j++) {
            double intermediateVal = 0;
            for(int i = 0; i < numberOfNeurons; i++)
                intermediateVal += weights[j][i] * hErrors[i];
            prevErrors[j] = intermediateVal * (prevLayer.neurons[j]) * (1 - prevLayer.neurons[j]);
        }

        /* Calculate deltas to update weights */
        /* Δ = η * δO * hi(E) */
        for(int j = 0; j < prevLayer.numberOfNeurons; j++)
            for(int i = 0; i < numberOfNeurons; i++) {
                double delta = learningRate * hErrors[i] * prevLayer.neurons[j];
                weights[j][i] += delta;
            }
        
        for(int i = 0; i < numberOfNeurons; i++) {
            double biasDelta = learningRate * hErrors[i];
            biases[i] = biasDelta;
        }
        prevLayer.backPropagation(prevErrors);
    }

    public String toString(char[] opcode) {
        String output = "HIDDEN LAYER " + index + '\n';
        if (opcode[0] == '1') {
            for (int i = 0; i < numberOfNeurons; ++i)
                output += "Neuron " + i + " : " + dfh.format(neurons[i]) + '\n';
            if (opcode[1] == '1')
                output += '\n';
        }
        if (opcode[1] == '1') {
            for (int i = 0; i < numberOfNeurons; ++i)
                output += "Bias " + i + " : " + dfh.format(biases[i]) + '\n';
            if (opcode[2] == '1')
                output += '\n';
        }
        if (opcode[2] == '1')
            for (int j = 0; j < prevLayer.numberOfNeurons; ++j) {
                for (int i = 0; i < numberOfNeurons; ++i)
                    output += "Weight " + j + "->" + i + " : " + dfh.format(weights[j][i]) + '\n';
                output += '\n';
            }
        return output;
    }
}

/**
 * Class responsible for representing the last layer of the neural network, extending HiddenLayer and using its properties
 */
class OutputLayer extends HiddenLayer {

    String[] names;
    public boolean stop;
    
    public OutputLayer(int numberOfNeurons, Layer prevLayer, HashSet<String> possibleOutputs, double learningRate) {
        super(numberOfNeurons, prevLayer, -1, learningRate);
        names = new String[numberOfNeurons];
        int i = 0;
        for(String name: possibleOutputs) {
            names[i] = name;
            i++;
        }
    }   

    public void backPropagation(int correctIndex) {
        /* 1 for the correct output, 0 for everything else */
        int[] target = new int[numberOfNeurons];
        fillTarget(target, correctIndex);

        /* Calculate output errors (δO) */
        /* δOi = oi(E)(1 - oi(E))(t1(E) - oi(E)) */
        double[] outputErrors = new double[numberOfNeurons];
        for(int i = 0; i < numberOfNeurons; i++) {
            outputErrors[i] = neurons[i] * (1 - neurons[i]) * (target[i] - neurons[i]);
            if(target[i] == 1) 
                stop = neurons[i] >= 0.95 ? true : false;   
        }
    
        /* Calculate errors to propagate to the hidden layers (δH) */
        double[] prevErrors = new double[prevLayer.numberOfNeurons];
        /* j is the previous layer neuron index */
        for(int j = 0; j < prevLayer.numberOfNeurons; j++) {
            double intermediateVal = 0;
            for(int i = 0; i < numberOfNeurons; i++) {
                double val = weights[j][i] * outputErrors[i];
                intermediateVal += val;
            }

            /* δHj */
            prevErrors[j] = intermediateVal * (prevLayer.neurons[j]) * (1 - prevLayer.neurons[j]);
        }

        /* Calculate deltas to update weights */
        for(int j = 0; j < prevLayer.numberOfNeurons; j++)
            for(int i = 0; i < numberOfNeurons; i++) {
                double weightDelta = learningRate * outputErrors[i] * prevLayer.neurons[j];
                weights[j][i] += weightDelta;
            }

        for(int i = 0; i < numberOfNeurons; i++) {
            double biasDelta = learningRate * outputErrors[i];
            biases[i] = biasDelta;
        }
        prevLayer.backPropagation(prevErrors);
    }
    
    /* Returns the name of the neuron with the highest value */
    public String getResult() {
        double max = 0;
        String result = "";
        for(int i = 0; i < numberOfNeurons; i++)
            if(neurons[i] > max) {
                max = neurons[i];
                result = names[i];
            }
        return result;
    }

    /* Returns the index of the neuron with the highest value */
    public int getBestIndex() {
        double max = 0;
        int index = -1;
        for(int i = 0; i < numberOfNeurons; i++)
            if(neurons[i] > max) {
                max = neurons[i];
                index = i;
            }
        return index;
    }

    /* Returns the index of the neuron with the given name */
    public int getCorrectIndex(String expectedVal) {
        for(int i = 0; i < numberOfNeurons; i++)
            if(expectedVal.equals(names[i]))
                return i;
        return -1;
    }

    /* Fill the target array for the current example (1 - expectedVal | 0 - everything else) */
    public static void fillTarget(int[] target, int correctIndex) {
        for(int i = 0; i < target.length; i++)
            target[i] = i == correctIndex ? 1 : 0;
    }

    public String toString(char[] opcode) {
        String output = "OUTPUT LAYER\n";
        if (opcode[0] == '1') {
            for (int i = 0; i < numberOfNeurons; ++i)
                output += names[i] + ": " + dfh.format(neurons[i]) + '\n';
            if (opcode[1] == '1')
                output += '\n';
        }
        if (opcode[1] == '1') {
            for (int i = 0; i < numberOfNeurons; ++i)
                output += "Bias " + i + ": " + dfh.format(biases[i]) + '\n';
            if (opcode[2] == '1')
                output += '\n';
        }
        if (opcode[2] == '1')
            for (int j = 0; j < prevLayer.numberOfNeurons; ++j) {
                for (int i = 0; i < numberOfNeurons; ++i)
                    output += "Weight " + j + "->" + i + ": " + dfh.format(weights[j][i]) + '\n';
                output += '\n';
            }
        return output;
    }
}

/**
 * Class responsible for just the first layer of the neural network
 */
class InputLayer extends Layer{

    public InputLayer(int numberOfNeurons) {
        super(numberOfNeurons);
    }

    public void setInputValues(double[] inputs) {
        for (int i = 0; i < numberOfNeurons; ++i)
            neurons[i] = inputs[i];
    }

    public void backPropagation(double[] dontCare) {}

    public String toString(char[] opcode) {
        String output = "";
        if (opcode[0] == '1') {
            output += "INPUT LAYER\n";
            for (int i = 0; i < numberOfNeurons; ++i)
                output += "Neuron " + i + " : " + neurons[i] + '\n';
        }
        return output;
    }
}