public class Runner
{
    public static void main(String[] args)
    {
        Network n          = new Network();
        double[][] inputs  = new double[][] {{.1, .1}, {.2, .2}, {.15, .15}, {.4, .5}};
        double[][] outputs = new double[][] {{.1, .1}, {.2, .2}, {.15, .15}, {.4, .5}};
        double[][] yhat    = n.passForward(inputs);
        double cost        = n.costFunction(outputs);

        System.out.println("Inputs:");
        Utility.printMat(inputs);
        System.out.println("\nOutputs:");
        Utility.printMat(outputs);
        System.out.println("\nY Hat:");
        Utility.printMat(yhat);
        System.out.println("\nCost:" + cost);
        System.out.println("Weights 0:");
        Utility.printMat(n.weights.get(0));
        System.out.println("\nWeights 1:");
        Utility.printMat(n.weights.get(1));

        double[][] layer2 = GradientDescent.gradientDescent(n, outputs, 2);
        double[][] layer1 = GradientDescent.gradientDescent(n, outputs, 1);

        System.out.println("\nError 1:");
        Utility.printMat(layer1);
        System.out.println("\nError 2:");
        Utility.printMat(layer2);

        n.subtractFromWeights(layer1, 0);
        n.subtractFromWeights(layer2, 1);

        for (int i = 0; i < 1000000; i++)
        {
            if (i % 100000 == 0)
            {
                cost = n.costFunction(outputs);
                System.out.println("Cost:" + cost);
            }
            n.passForward(inputs);
            layer1 = GradientDescent.gradientDescent(n, outputs, 1);
            layer2 = GradientDescent.gradientDescent(n, outputs, 2);
            n.subtractFromWeights(layer1, 0);
            n.subtractFromWeights(layer2, 1);
        }

        yhat = n.passForward(inputs);
        cost = n.costFunction(outputs);

        System.out.println("\nInputs:");
        Utility.printMat(inputs);
        System.out.println("\nOutputs:");
        Utility.printMat(outputs);
        System.out.println("\nY Hat:");
        Utility.printMat(yhat);
        System.out.println("\nCost:" + cost);
        System.out.println("Weights 0:");
        Utility.printMat(n.weights.get(0));
        System.out.println("\nWeights 1:");
        Utility.printMat(n.weights.get(1));
        System.out.println();

        yhat = n.passForward(new double[][] {{.05, .05}});
        Utility.printMat(yhat);
        yhat = n.passForward(new double[][] {{.1, .2}});
        Utility.printMat(yhat);
    }
}
