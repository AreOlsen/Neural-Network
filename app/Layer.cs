using System;
namespace NeuralNetwork{
    public class Layer{
        public int numberInputNuerons,numberNeurons;
        public double[,] weights;
        public double[] biases;
        public double[,] costGradientWeight;
        public double[] costGradientBias;

        /* Populating the layer using the amount of nodes going in and the layer's own nueron count. */
        public Layer(int numberInputNuerons, int numberNeurons){
            this.numberInputNuerons=numberInputNuerons;
            this.numberNeurons=numberNeurons;
            weights = new double[numberInputNuerons,numberNeurons]; /* Weights into and weights out-*/
            biases = new double[numberNeurons]; /* Current number of biases for layer. */
            InitWeights();
        }
        /*
            Layer Output.
        */
        public double[] CalculateTheOutPut(double[] inputs){
            double[] outputs = new double[numberNeurons];
            /* Go through each nueron in layer.*/
            for(int nueronIndex = 0; nueronIndex < numberNeurons; nueronIndex++){
                /* 
                    Get the current bias for the the current nueron.
                    Instead of appending all the biases at the end we can just start with them and add onto.
                */
                double outValue = biases[nueronIndex];
                for(int inputIndex = 0; inputIndex < numberInputNuerons; inputIndex++){
                    /*
                     For each in going nueron get the input value and use the weight on it.
                    The output is a calculation of all the inputs.
                    */
                    outValue+=inputs[inputIndex]*weights[inputIndex,nueronIndex];
                } 
                /* We say that the current nueron's output value is all the inputs' values with their corresponding weights
                 run through an activation function. We also have the bias taken intot the activation function, it was set at first.
                */
                outputs[nueronIndex]=Functions.ReLU.Calculate(outValue);
            }
            return outputs;
        }

        /*
            Apply the weights and biases based on the costs (Gradient Descent)
        */
        public void ApplyGradient(double learnRate){
            for(int nueronI=0;nueronI<numberNeurons; nueronI++){
                biases[nueronI]-=costGradientBias[nueronI]*learnRate;
                for(int neuronInI = 0; neuronInI<numberInputNuerons; neuronInI++){
                    weights[neuronInI,nueronI]=costGradientWeight[neuronInI,nueronI]*learnRate;
                }
            }
        }

        public void UpdateGradients(double[] nodeValues){
            for(int nueronI =0; nueronI<numberNeurons; nueronI++){
                for(int nueronInI = 0; nueronInI<numberInputNuerons;nueronInI++){

                    double costDerivativeWeight=inputs[nueronInI]*nodeValues[nueronI];
                    costGradientWeight[nueronInI,nueronI]+=costDerivativeWeight;
                }
                double costDerivativeBias = nodeValues[nueronI];
                costGradientBias[nueronI]+=costDerivativeBias;
            }
        }





        /* 
            There are a lot of different mathematical theories around the best weight range.
            However there are only two known facts for certain:
            1. Weights normally shouldnt be extreme in init.
            2. Weights have to be random at init.
            Not so good facts!
            This is a simple init function.
            Different types of networks have different "close to perfect" ones: This is a good one for sigmoid.
        */
        public void InitWeights(){
            System.Random rand = new System.Random();
            for(int nueronIniI= 0; nueronIniI<numberInputNuerons; nueronIniI++){
                for(int neuronI = 0; neuronI<numberNeurons; neuronI++){
                     double random = rand.NextDouble()*2-1; /* -1 between 1 in value*/
                     weights[nueronIniI,neuronI]=random/Math.Sqrt(numberInputNuerons);
                }
            }
        }

        /* 
            Node Cost Function.
        */
        public double CostNueron(double activedOutput, double expectedOutput){
            double error = expectedOutput-activedOutput; /* In decimal ofc. */
            return error*error; /* This amplifies the effect of the error. Makes small errors act gravely terrible! */
        }
        public double CostNueronDerivative(double activedOutput, double expectedOutput){
            return 2*(expectedOutput-activedOutput);
        }

        public double[] CalculateOutputLayerNodeVals(double[] expectedValues){
            double[] nodeVals = new double[expectedValues.Length];
            for(int i = 0; i< nodeVals.Length; i++){
                // Eval partial derivs. for current node. cost/activation & activation/weightedinput.
                double costDerivative = CostNueronDerivative(activations[i],expectedValues[i]);
                double activationDerivative = Functions.ReLU.Calculate(weightedInputs[i]);
                nodeVals[i] = activationDerivative*costDerivative;
            }
            return nodeVals;
        }

        public double[] CalculateHiddenLayerNodeVals(Layer oldLayer, double[] oldNodeVals){
            double[] newNodeVals = new double[numberNeurons];
            for(int newNodeI = 0; newNodeI<newNodeVals.Length; newNodeI++){
                double newNodeVal = 0;
                for(int oldNodeI = 0; oldNodeI < oldNodeVals.Length; oldNodeI++){
                    //Partial deriv. of the weighte input with respect to the given input.
                    double weightedInputDerivative = oldLayer.weights[newNodeI,oldNodeI];
                    newNodeVal+=weightedInputDerivative*oldNodeVals[oldNodeI];
                }
                newNodeVal+=Functions.ReLU.Calculate(weightedInputs[newNodeI]);
                newNodeVals[newNodeI]=newNodeVal;
            }
            return newNodeVals;
        }

        public void ClearGradient(){
            for(int i = 0; i < costGradientBias.Length; i++){
                costGradientBias[i]=0;
            }
            for(int i = 0; i < costGradientWeight.Length; i++){
                costGradientWeight[i]=0;
            }
        }
    }
}