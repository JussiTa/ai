package digia.ai.com;

import java.io.IOException;
import java.nio.file.Paths;

import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.optimisers.AdaGrad;
import java.util.*;



public class App 
{
    public static void main( String[] args ) throws IOException
    {
   
      
      LabelFactory    labelFactory = new LabelFactory();
      try {
        var trainDataSource = new IDXDataSource<>(Paths.get("src/main/resources/train-images-idx3-ubyte.gz"), Paths.get("src/main/resources/train-labels-idx1-ubyte.gz"), labelFactory);
       
        var trainDataset = new MutableDataset<>(trainDataSource);

        var trainer = new LinearSGDTrainer(new LogMulticlass(), new AdaGrad(0.5), 5, 42);
        
        var model = trainer.train(trainDataset);
        System.out.println(model.toString());
        
        var testDataSource = new IDXDataSource<>(Paths.get("src/main/resources/t10k-images-idx3-ubyte.gz"), Paths.get("src/main/resources/t10k-labels-idx1-ubyte.gz"), labelFactory);
        Prediction<Label> prediction = model.predict(testDataSource.iterator().next());
        System.out.println(prediction.getExample());
        List<Prediction<Label>> batchPredictions = model.predict(testDataSource);

        var evaluator = new LabelEvaluator();
        System.out.println(testDataSource.getDataType());
        var evaluation = evaluator.evaluate(model, batchPredictions, testDataSource.getProvenance());

        System.out.println(evaluation.toString());

        
        
        
    
    } catch (IOException e) {
        // TODO Auto-generated catch block
        e.printStackTrace(); 

    }




        // Load labelled iris data

    }
    


}
