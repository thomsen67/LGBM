using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
//using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using LightGBMNet.Tree;

public class IrisInput
{
    // features exactly match LightGBM model feature order
    public float Column_0 { get; set; }
    public float Column_1 { get; set; }
    public uint Column_2 { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        Program2.Xxx(); return;
        Train.Yyy();
    }
}

public class Program2
{
    public static void Xxx()
    {
        // 1. Load the model using Ensemble.Parse
        // ReadLines is better than ReadAllLines for memory, 
        // as Parse iterates through the lines.
        (Ensemble ensemble, Parameters p, int i) = Ensemble.GetModelFromString(File.ReadAllText(@"c:\Thomas\Desktop\gekko\testing\DREAM\LGBM_TTH_Repo\adult_model.txt"));

        // 2. Prepare your input features (14 features based on your file)
        // Must follow the order: age, workclass, fnlwgt, education, etc.
        float[] inputFeatures = new float[]
        {
            39f, 7f, 77516f, 9f, 13f, 4f, 1f, 1f, 4f, 1f, 2174f, 0f, 40f, 38f
        };



        // 3. Wrap the array in a VBuffer
        // VBuffer constructor: (length, array)
        var vBuffer = new VBuffer<float>(inputFeatures.Length, inputFeatures);

        // 4. Call GetOutput with the required signature
        // startIteration: 0
        // numIterations: ensemble.NumTrees (or the count of trees in the list)
        int numTrees = ensemble.Trees.Count();// Count;
        double rawScore = ensemble.GetOutput(ref vBuffer, 0, numTrees);

        // 4. Since your file says "objective=binary sigmoid", 
        // convert the logit to a probability.
        double probability = Sigmoid(rawScore);

        Console.WriteLine($"Raw Score: {rawScore}");
        Console.WriteLine($"Probability: {probability:P2}");

    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}