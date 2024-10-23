{
    "imagesFolder": "/Users/hugom/Tesis/Imagenes/ADNI-MUESTRA-FULL-stripped-preprocessed3",
    "fleniImagesFolder": "/Users/hugom/Tesis/Imagenes/fleni-stripped-preprocessed3",
    "trainDatasetCSV": "../Sets/Muestra3700_80_10_10_dxlast_train.csv",
    "valDatasetCSV": "../Sets/Muestra3700_80_10_10_dxlast_test.csv",
    "fleniValDatasetCSV": "../Sets/fleni-myriam-curated.csv",
    "experimentName": "muestraFull3700_10_dxlast_3classes",
    "experimentOutputFolder": "/Users/hugom/Tesis/ExperimentosServer/muestraFull3700_10_test",
    "experimentDescription": "Experimento usando el dxlast con 3 clases",
    "executions": 1,
    "model_name": "inception",
    "num_classes": 3,
    "batch_size": 8,
    "dl_num_workers": 0,
    "num_epochs": 50,
    "feature_extract": false,
    "usePretrained": true,
    "auxEnabled": true,
    "learningRate": 0.0001,
    "dropoutRate": 0.6,
    "trainElements": [ 765.0, 1039.0, 1229.0 ],
    "trainMean": 0.26114351157369636,
    "trainStd": 0.46353330246995716,
    "fleniMean": 3364.6066073463076,
    "fleniStd": 7271.672596534478,
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
	"processStatsADNI": true,
	"processStatsFleni": false
    },
    "truthLabel": "DX_last"
}
