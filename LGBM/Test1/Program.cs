using LightGBMNet.Tree;

class Program
{
    static void Main(string[] args)
    {
        (Ensemble ensemble, Parameters p, int i) = Ensemble.GetModelFromString(File.ReadAllText(@"c:\Thomas\Desktop\gekko\testing\DREAM\LGBM_TTH_Repo\adult_model.txt"));
        float[] inputFeatures = new float[]
        {
            39f, 7f, 77516f, 9f, 13f, 4f, 1f, 1f, 4f, 1f, 2174f, 0f, 40f, 38f
        };
        VBuffer<float> vBuffer = new VBuffer<float>(inputFeatures.Length, inputFeatures);
        double rawScore = ensemble.GetOutput(ref vBuffer, 0, ensemble.Trees.Count());
        double probability = Sigmoid(rawScore);  //since it is sigmoid
        Console.WriteLine($"Raw Score: {rawScore}");
        Console.WriteLine($"Probability: {probability:P2}");
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}
