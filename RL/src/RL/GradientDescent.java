package RL;

public class GradientDescent {

    public float costFunction(float[] yHat, float[] y)
    {
        if (y.length != yHat.length) return -1;
        float sum = 0;
        for (int i = 0; i < y.length; i++)
        {
            float error = (float) Math.pow(y[i] - yHat[i], 2);
            sum += error;
        }
        return sum / 2;
    }

    public float dJdW(float input, float output, float cost)
    {
        float yminyhat = output - cost;
        return 0;
    }
}
