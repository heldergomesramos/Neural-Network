import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Locale;
import java.util.Scanner;

public class Main {

    public static Scanner in = new Scanner(System.in).useLocale(Locale.US);
    public static long timeLimit = 7000;
    public static void main(String[] args) throws IOException {

        if(args.length != 1 && args.length != 2)
            throw new IOException("Wrong number of arguments!");

        ArrayList<double[]> inputMatrix = new ArrayList<>();
        ArrayList<String> expectedValues = new ArrayList<>();
        HashSet<String> possibleOutputs = new HashSet<>();
        ArrayList<double[]> testMatrix = new ArrayList<>();
        ArrayList<String> idList = new ArrayList<>();

        readTrainingData(args[0], inputMatrix, expectedValues, possibleOutputs);
        NeuralNetwork nn = NeuralNetwork.inputNeuralNetwork(possibleOutputs, inputMatrix.get(0).length, in);
        train(nn, inputMatrix, expectedValues);

        char[] opcode = {'0','1','1'};
        nn.debug(opcode);

        if(args.length == 2) {
            readTestData(args[1], idList, testMatrix);
            test(nn, idList, testMatrix);
        }
    }

    public static void readTrainingData(String arg, ArrayList<double[]> inputMatrix, ArrayList<String> expectedValues, HashSet<String> possibleOutputs) throws IOException {
        FileReader input = new FileReader(arg);
        BufferedReader csvFile = new BufferedReader(input);
        String inputLine = csvFile.readLine();

        /* For each line */
        while(inputLine != null) {
            String[] splittedLine = inputLine.split(",");
            /* inputExample is the splittedLine without the expected value */
            double[] inputExample = new double[splittedLine.length - 1];
            for(int i = 0; i < splittedLine.length - 1; i++)
                inputExample[i] = Double.parseDouble(splittedLine[i]);
            inputMatrix.add(inputExample);
            expectedValues.add(splittedLine[splittedLine.length - 1]);
            possibleOutputs.add(splittedLine[splittedLine.length - 1]);
            inputLine = csvFile.readLine();
        }
        csvFile.close();
    }

    public static void readTestData(String arg, ArrayList<String> idList, ArrayList<double[]> testMatrix) throws IOException {
        FileReader input = new FileReader(arg);
        BufferedReader csvFile = new BufferedReader(input);
        String inputLine = csvFile.readLine();

        /* For each line */
        while(inputLine != null) {
            String[] splittedLine = inputLine.split(",");
            double[] inputTest = new double[splittedLine.length - 1];
            idList.add(splittedLine[0]);
            for(int i = 1; i < splittedLine.length; i++)
                inputTest[i - 1] = Double.parseDouble(splittedLine[i]);
            testMatrix.add(inputTest);
            inputLine = csvFile.readLine();
        }
        csvFile.close();
    }

    public static void train(NeuralNetwork nn, ArrayList<double[]> inputMatrix, ArrayList<String> expectedValues) {
        System.out.println("TRAINING...");
        long startTime = System.currentTimeMillis();
        long timeElapsed = startTime;
        boolean stop = false;
        int iterations = 0;
        double successRatio = 0;
        for(; stop == false; iterations++) {
            stop = true;
            double hits = 0;
            for(int i = 0; i < inputMatrix.size(); i++) {
                String expectedVal = expectedValues.get(i);
                hits += nn.trainNetwork(inputMatrix.get(i), expectedVal);
                if(!nn.outputLayer.stop)
                    stop = false;
            }
            timeElapsed = System.currentTimeMillis() - startTime;
            successRatio = hits / inputMatrix.size();
            if(timeElapsed >= timeLimit) {
                System.out.println("Time limit exceeded: " + timeLimit + " ms");
                break;
            }
        }
        System.out.println("Training finished with " + iterations + " iterations");
        DecimalFormat df3 = new DecimalFormat("00.0");
        System.out.println("Success ratio with training samples: " + df3.format(successRatio * 100) + "%");
    }

    public static void test(NeuralNetwork nn, ArrayList<String> idList, ArrayList<double[]> testMatrix) {
        for (int i = 0; i < idList.size(); i++) {
            String result = nn.test(testMatrix.get(i));
            System.out.println(idList.get(i) + " - " + result);
        }
    }
}
