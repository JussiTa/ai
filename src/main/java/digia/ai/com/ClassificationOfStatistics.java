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


public class ClassificationOfStatistics{


    public static void main(String args []) throws IOException{

        var statHeaders = new String[]{"Variance","Skewness","Curtosis","Entropy","Class"};

        DataSource<Label> staticData = new CSVLoader<>(new LabelFactory()).
        loadDataSource(Paths.get("src/main/resources/bill_authentication.csv"), statHeaders[4] , statHeaders);

        

        var splitStatisticsData = new TrainTestSplitter<>(staticData, 0.8, 1L);

        var trainData = new MutableDataset<>((splitStatisticsData.getTrain()));
        var testData = new MutableDataset<>(splitStatisticsData.getTest());

        var cartTrainer = new CARTClassificationTrainer();
        Model<Label> tree = cartTrainer.train(trainData);
       
        
        var linearTriner = new LogisticRegressionTrainer();
        Model<Label> linear = linearTriner.train(trainData);
        
        Prediction<Label> prediction = linear.predict(testData.getExample(1));
        

        System.out.println(testData.toString());
        LabelEvaluation eval = new LabelEvaluator().evaluate(linear, testData);
        
        System.out.println(eval.toString());

  
      }


    }





