using System;

namespace NeuralNetwork {
    public class Network {

        Layer[] layers;
        float learnRate;
        public Network( int[] hiddenLayer, int inputLayerNeuronNumber, int outputLayerNeuronNumber, float learnRate = 0.01f){
            this.learnRate=learnRate;
            layers = new Layer[hiddenLayer.Length+2];
            layers[0] = new Layer(0,inputLayerNeuronNumber);
            layers[1] = new Layer(inputLayerNeuronNumber, hiddenLayer[0]);
            for(int i = 1; i<hiddenLayer.Length-1; i++){
                layers[i]=new Layer(hiddenLayer[i-1],hiddenLayer[i]);
            }
            layers[hiddenLayer.Length] = new Layer(hiddenLayer[hiddenLayer.Length], outputLayerNeuronNumber); /* Second last. */
            layers[hiddenLayer.Length+1]= new Layer(hiddenLayer.Length,0); /* Last output layer*/
        }

        public double[] ForwardPropagation(double[] inputs){
            /* Go forward through the layers and calculate the output. */
            foreach(Layer layer in layers){
                inputs = layer.CalculateTheOutPut(inputs);
            }
            return inputs;
        }

        public int Classify(double[] inputs){
            /* Get the outputs.*/
            double[] outputs = ForwardPropagation(inputs);
            /* Find index of highest value. Can be changed to just return the value.*/
            int index = -1;
            if(outputs != null && outputs.Length > 0) {
                index = Array.IndexOf(outputs,outputs.Max());
            }
            return index;
        }

        public double Cost(InputPoint dataPoint){ /* Loss for one datapoint aka input-neuron values. */
            double[] outputs = ForwardPropagation(dataPoint.inputs);
            Layer outputLayer = layers[layers.Length-1];
            double loss = 0;
            /* Go through all outputs give a loss value telling us the quality of calculations. */
            for(int i = 0; i < outputs.Length; i++){
                loss+=outputLayer.CostNueron(outputs[i],dataPoint.expectOutputs[i]);
            }
            return loss;
            /*
                This is the Mean Square Loss function. 
            */
        }


        /* Get total loss.
            We want the total loss to be minimum.
            This is called learning!        
        */
        public double Loss(InputPoint[] data){ 
            double totalLoss = 0;
            foreach(InputPoint datapoint in data){
                totalLoss+=Cost(datapoint);
            }
            return totalLoss/data.Length;
        }
        
        /* Single Gradient Descent iteration. */
        public void Learn(InputPoint[] trainingData, double learnRate){
            /* 
                Using backwards propagation we calculate gradient of cost function.
                With respect to network weights and biases.
                Done for each inputpoint / trainingdata point and added together.
            */
            foreach(InputPoint input in trainingData){
                    UpdateAllGradients(input);
            }
            /* Gradient desecnt step- update all weights and biases in network.*/
            foreach(Layer layer in layers){
                layer.ApplyGradient(learnRate/trainingData.Length);
            }
            /* Clear all gradients to be ready for next traing batch. */
            foreach(Layer layer in layers){
                layer.ClearGradient();
            }
        }
        

        /* 
            Backwards-Propagation Algorithm.
        */
        public void UpdateAllGradients(InputPoint input){
            /* Run data through the layers. The layers will store needed values.*/
            ForwardPropagation(input.inputs);
            Layer outputLayer = layers[layers.Length-1];
            double[] nodeVals = outputLayer.CalculateOutputLayerNodeVals(input.expectOutputs);
            outputLayer.UpdateGradients(nodeVals);
            /* Loop through all hidden layers - update their gradients. */
            for(int hiddenLayerIndex = layers.Length-2; hiddenLayerIndex>=0; hiddenLayerIndex--){
                Layer hiddenLayer = layers[hiddenLayerIndex];
                nodeVals = hiddenLayer.CalculateHiddenLayerNodeVals(layers[hiddenLayerIndex+1],nodeVals);
                hiddenLayer.UpdateGradients(nodeVals);
            }
        }
    }
}
