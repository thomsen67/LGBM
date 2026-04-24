using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

public class IrisInput
{
    // features exactly match LightGBM model feature order
    public float SepalLength { get; set; }
    public float SepalWidth { get; set; }
    public float PetalLength { get; set; }
    public float PetalWidth { get; set; }
}

public class IrisPrediction
{
    // multiclass LightGBM outputs Score
    [ColumnName("Score")]
    public float[] Score { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        Program2.Xxx(); return;
        
        // 1) Create ML context
        var mlContext = new MLContext();

        // 2) Load pretrained model file as stream
        using var modelStream = File.OpenRead(@"c:\Thomas\Desktop\gekko\testing\DREAM\LightGBM\CSharp3\Test\iris.txt");

        var pipeline = mlContext.Transforms.Concatenate(
        "Features",
        nameof(IrisInput.SepalLength),
        nameof(IrisInput.SepalWidth),
        nameof(IrisInput.PetalLength),
        nameof(IrisInput.PetalWidth)
    )
    .Append(mlContext.MulticlassClassification.Trainers
        .LightGbm(modelStream, featureColumnName: "Features"));




        // 3) Build an empty pipeline that uses the pretrained LightGBM
        //    Note: featureColumn must be a vector named "Features"
        var pipeline2 = mlContext.Transforms.Concatenate(
                "Features",
                nameof(IrisInput.SepalLength),
                nameof(IrisInput.SepalWidth),
                nameof(IrisInput.PetalLength),
                nameof(IrisInput.PetalWidth)
            )
            // Here we plug in the pretrained model stream
            .Append(mlContext.MulticlassClassification.Trainers
                .LightGbm(modelStream, featureColumnName: "Features"))
            // ML.NET multiclass needs a key-to-value map for labels if needed
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // 4) Load *some* empty data just to create a DataView
        var empty = new List<IrisInput>();
        var emptyDv = mlContext.Data.LoadFromEnumerable(empty);

        // 5) Fit — this attaches the pretrained model to the pipeline
        var mlModel = pipeline.Fit(emptyDv);

        // 6) Create prediction engine
        var predictionEngine =
            mlContext.Model.CreatePredictionEngine<IrisInput, IrisPrediction>(mlModel);

        // 7) Predict on sample
        var sample = new IrisInput
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };


        var prediction = predictionEngine.Predict(sample);
        var scores = prediction.Score;
        var predictedClassIndex = Array.IndexOf(scores, scores.Max());
        Console.WriteLine($"Predicted class index: {predictedClassIndex}");

        //var prediction = predictionEngine.Predict(sample);

        Console.WriteLine("Predicted class scores:");
        for (int i = 0; i < prediction.Score.Length; i++)
        {
            Console.WriteLine($"Class {i}: {prediction.Score[i]:F4}");
        }
    }
}

public class Program2
{
    public static void Xxx()
    {
        var mlContext = new MLContext();

        // 1. Empty data WITH correct schema matching the features below
        var emptyData = mlContext.Data.LoadFromEnumerable(new List<IrisInput>());

        // 2. Build full pipeline: Concatenate + pretrained LightGBM
        var modelPath = @"c:\Thomas\Desktop\gekko\testing\DREAM\LightGBM\CSharp3\Test\iris.txt";

        using var modelStream = File.OpenRead(modelPath);

        var pipeline = mlContext.Transforms.Concatenate(
                "Features",
                nameof(IrisInput.SepalLength),
                nameof(IrisInput.SepalWidth),
                nameof(IrisInput.PetalLength),
                nameof(IrisInput.PetalWidth)
            )
            .Append(mlContext.MulticlassClassification.Trainers
                .LightGbm(modelStream, featureColumnName: "Features"));

        // 3. Fit (loads pretrained weights, no actual training)
        var fittedModel = pipeline.Fit(emptyData);

        // 4. Save as .zip — schema comes from emptyData, model from fittedModel
        var savePath = @"c:\Thomas\Desktop\gekko\testing\DREAM\LightGBM\CSharp3\Test\iris.zip";
        mlContext.Model.Save(fittedModel, emptyData.Schema, savePath);

        Console.WriteLine($"Model saved to {savePath}");

        // 5. Verify: reload and predict
        var loadedModel = mlContext.Model.Load(savePath, out var schema);
        var predEngine = mlContext.Model
            .CreatePredictionEngine<IrisInput, IrisPrediction>(loadedModel);

        var result = predEngine.Predict(new IrisInput
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        });

        Console.WriteLine("Scores: " + string.Join(", ",
            Array.ConvertAll(result.Score, s => s.ToString("F4"))));

    }
}