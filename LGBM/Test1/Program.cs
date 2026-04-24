using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

public class IrisInput
{
    // features exactly match LightGBM model feature order
    public float Column_0 { get; set; }
    public float Column_1 { get; set; }
    public uint Column_2 { get; set; }
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
        //Program2.Xxx(); return;
        Train.Yyy();
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
        var modelPath = @"c:\Thomas\Desktop\gekko\testing\DREAM\LGBM_TTH_Repo\lgbm.txt";

        using var modelStream = File.OpenRead(modelPath);

        var pipeline = mlContext.Transforms.Concatenate(
                "Features",
                nameof(IrisInput.Column_0),
                nameof(IrisInput.Column_1),
                nameof(IrisInput.Column_2)
            )
            .Append(mlContext.MulticlassClassification.Trainers
                .LightGbm(modelStream, featureColumnName: "Features"));

        // 3. Fit (loads pretrained weights, no actual training)
        var fittedModel = pipeline.Fit(emptyData);

        // 4. Save as .zip — schema comes from emptyData, model from fittedModel
        var savePath = @"c:\Thomas\Desktop\gekko\testing\DREAM\LGBM_TTH_Repo\lgbm.zip";
        mlContext.Model.Save(fittedModel, emptyData.Schema, savePath);

        Console.WriteLine($"Model saved to {savePath}");

        // 5. Verify: reload and predict
        var loadedModel = mlContext.Model.Load(savePath, out var schema);
        var predEngine = mlContext.Model
            .CreatePredictionEngine<IrisInput, IrisPrediction>(loadedModel);

        var result = predEngine.Predict(new IrisInput
        {
            Column_0= 5.1f,
            Column_1 = 3.5f,
            Column_2 = 2
        });

        Console.WriteLine("Scores: " + string.Join(", ",
            Array.ConvertAll(result.Score, s => s.ToString("F4"))));

    }
}