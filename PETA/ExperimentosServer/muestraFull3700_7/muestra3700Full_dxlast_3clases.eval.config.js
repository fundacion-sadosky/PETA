{
    "imagesFolder": "/Users/hugom/Tesis/Imagenes/ADNI-MUESTRA-FULL-stripped-preprocessed3",
    "fleniImagesFolder": "/Users/hugom/Tesis/Imagenes/fleni-stripped-preprocessed3",
    "trainDatasetCSV": "../Sets/Muestra3700_80_10_10_dxlast_train.csv",
    "valDatasetCSV": "../Sets/Muestra3700_80_10_10_dxlast_val.csv",
    "fleniValDatasetCSV": "../Sets/fleni-myriam-curated.csv",
    "experimentName": "muestraFull3700_7_dxlast_3classes",
    "experimentOutputFolder": "/Users/hugom/Tesis/ExperimentosServer/muestraFull3700_7",
    "experimentDescription": "Experimento usando el dxlast con 3 clases",
    "executions": 1,
    "model_name": "inception",
    "num_classes": 3,
    "batch_size": 32,
    "dl_num_workers": 0,
    "num_epochs": 50,
    "feature_extract": false,
    "usePretrained": true,
    "auxEnabled": true,
    "learningRate": 0.0001,
    "dropoutRate": 0.6,
    "trainElements": [ 765.0, 1039.0, 1229.0 ],
    "trainMean": 0.17155274981780977,
    "trainStd": 0.39358714727689015,
    "fleniMean": 2118.845256014825,
    "fleniStd": 5994.702274256329,
    "deviceName": "cpu",
    "dataAugmentation": {
        "angle": 8,
        "shiftX": 10,
        "shiftY": 10,
        "zoom": 0.1,
        "shear": 0.19634954084936207
    },
    "selectCriteria": "accuracy",
    "validationCacheSize": 300,
    "trainCacheSize": 0,
    "calculateAUCROC": true,
    "debug": false,
    "doTrain": true,
    "selectCriteriaAbbrv": {
        "accuracy": "acc",
        "f1AD": "f1AD"
    },
    "eval": {
	"processStatsADNI": false,
	"processStatsFleni": true
    },
    "truthLabel": "DX_last"
}
