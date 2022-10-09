using System.Collections.Generic;
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace AI
{
    /// <summary>
    /// Supported "activation" functions for the neuron layer.
    /// </summary>
    public enum ActivationFunctions { Sigmoid, TanH, ReLU, LeakyReLU };

    /// <summary>
    /// Implementation of a feedforward neural network.
    /// </summary>
    public class NeuralNetwork
    {
        #region ACTIVATION FUNCTIONS
        /// <summary>
        /// Delegate for activation functions.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private delegate double ActivationFunction(double input);

        /// <summary>
        /// Delegate for derivative of activation functions.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private delegate double ActivationDerivativeFunction(double input);

        /// <summary>
        /// Activation functions to be user per layer.
        /// </summary>
        private readonly ActivationFunction[] activationMethod;

        /// <summary>
        /// Activation derivative functions to be user per layer.
        /// </summary>
        private readonly ActivationDerivativeFunction[] activationDerivativeMethod;
        #endregion
        
        /// <summary>
        /// How many layers of neurons (2+). Do not do 1.
        /// 2 => input connected to output.
        /// 1 => input is output, and feed forward will crash.
        /// </summary>
        internal readonly int[] Layers;

        /// <summary>
        /// The neurons.
        /// [layer][neuron]
        /// </summary>
        internal double[][] Neurons;

        /// <summary>
        /// NN Biases. Either improves or lowers the chance of this neuron fully firing.
        /// [layer][neuron]
        /// </summary>
        private double[][] Biases;

        /// <summary>
        /// NN weights. Reduces or amplifies the output for the relationship between neurons in each layer
        /// [layer][neuron][neuron]
        /// </summary>
        private double[][][] Weights;

        /// <summary>
        /// LeakyReLU alpha.
        /// </summary>
        private const float alpha = 0.01f;

        #region BACK PROPAGATION
        /// <summary>
        /// 
        /// </summary>
        private readonly float learningRate = 0.01f;
        #endregion

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="layerDefinition">Defines size of the layers.</param>
        /// <param name="func">Defines activation functions.</param>
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable. ERRONEOUS: POPULATED IN METHODS AT END OF METHOD
        internal NeuralNetwork( int[] layerDefinition, ActivationFunctions[] func)
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
        {
            // (1) INPUT (2) HIDDEN (3) OUTPUT.
            if (layerDefinition.Length < 2) throw new ArgumentException(nameof(layerDefinition)+" insufficient layers.");

            // copy layerDefinition to Layers; although for cars this must not change.     
            Layers = new int[layerDefinition.Length];

            activationMethod = new ActivationFunction[layerDefinition.Length];
            activationDerivativeMethod = new ActivationDerivativeFunction[layerDefinition.Length];
            
            for (int layer = 0; layer < layerDefinition.Length; layer++)
            {
                Layers[layer] = layerDefinition[layer];

                GetActivationFunctions(func[layer], out ActivationFunction activationFunc,  out ActivationDerivativeFunction derivFunc);

                activationMethod[layer] = activationFunc;
                activationDerivativeMethod[layer] = derivFunc;
            }

            // if layerDefinition is [2,3,2] then...
            // 
            // Neurons :      (o) (o)    <-2  INPUT
            //              (o) (o) (o)  <-3
            //                (o) (o)    <-2  OUTPUT
            //

            InitialiseNeurons();
            InitialiseBiases();
            InitialiseWeights();
        }

        #region ACTIVATION / DERIVATIVE FUNCTIONS
        /// <summary>
        /// We assign a function pointer for both, to save resolving the activation functions at runtime.
        /// </summary>
        /// <param name="activationFunctions"></param>
        /// <param name="activationFunc"></param>
        /// <param name="derivativeActivationFunc"></param>
        /// <exception cref="NotImplementedException">You referenced an activation function that is unsupported.</exception>
        private void GetActivationFunctions(ActivationFunctions activationFunctions, out ActivationFunction activationFunc, out ActivationDerivativeFunction derivativeActivationFunc)
        {
            switch (activationFunctions)
            {
                case ActivationFunctions.Sigmoid:
                    activationFunc = SigmoidActivationFunction;
                    derivativeActivationFunc = DerivativeOfSigmoidDerivationFunction;
                    break;
                
                case ActivationFunctions.TanH:
                    activationFunc = TanHActivationFunction;
                    derivativeActivationFunc = DerivativeOfTanHActivationFunction;
                    break;
                
                case ActivationFunctions.ReLU:
                    activationFunc = ReLUActivationFunction;
                    derivativeActivationFunc = DerivativeOfReLUActivationFunction;
                    break;
                
                case ActivationFunctions.LeakyReLU:
                    activationFunc = LeakyReLUActivationFunction;
                    derivativeActivationFunc = DerivativeOfLeakyReLUActivationFunction;
                    break;

                default:
                    // if there is a missing function
                    throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear. 
        /// But unlike Sigmoid, its output is zero-centered. Therefore, in practice the tanh non-linearity is always preferred 
        /// to the sigmoid nonlinearity.
        /// 
        /// Activate is TANH         1_       ___
        /// (hyperbolic tangent)     0_      /
        ///                         -1_  ___/
        ///                                | | |
        ///                     -infinity -2 0 2..infinity
        ///                               
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        private static double TanHActivationFunction(double value)
        {
            return (double)Math.Tanh(value);
        }

        /// <summary>
        /// Derivative (for back-propagation of TanH activation function).
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double DerivativeOfTanHActivationFunction(double value)
        {
            return 1 - (value * value);
        }

        /// <summary>
        /// Sigmoid takes a real value as input and outputs another value between 0 and 1. 
        /// It’s easy to work with and has all the nice properties of activation functions: 
        /// it’s non-linear, continuously differentiable, monotonic, and has a fixed output range.
        /// 
        /// Pros
        /// - It is nonlinear in nature. Combinations of this function are also nonlinear!
        /// - It will give an analog activation unlike step function.
        /// - It has a smooth gradient too.
        /// - It’s good for a classifier.
        /// - The output of the activation function is always going to be in range (0,1) compared to(-inf, inf) of linear function.So we have our activations bound in a range.Nice, it won’t blow up the activations then.
        /// 
        /// Cons
        /// - Towards either end of the sigmoid function, the Y values tend to respond very less to changes in X.
        /// - It gives rise to a problem of “vanishing gradients”.
        /// - Its output isn’t zero centered.It makes the gradient updates go too far in different directions. 0 < output< 1, and it makes optimization harder.
        /// - Sigmoids saturate and kill gradients.
        /// - The network refuses to learn further or is drastically slow (depending on use case and until gradient /computation gets hit by floating point value limits).
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private static double SigmoidActivationFunction(double input)
        {
            double k = (double)Math.Exp(input);
            return k / (1.0f + k);
            // ?
            // def sigmoid(z):
            //   return 1.0 / (1 + np.exp(-z))
        }

        /// <summary>
        /// Derivative (for back-propagation of Sigmoid activation function).
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private static double DerivativeOfSigmoidDerivationFunction(double input)
        {
            return input * (1 - input);

            // ?
            // def sigmoid_prime(z):
            //   return sigmoid(z) * (1 - sigmoid(z))
        }

        /// <summary>
        /// The rectified linear activation function or ReLU for short is a piecewise linear function that will output 
        /// the input directly if it is positive, otherwise, it will output zero. It has become the default activation 
        /// function for many types of neural networks because a model that uses it is easier to train and often achieves 
        /// better performance.
        /// 
        /// See: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/#:~:text=The%20rectified%20linear%20activation%20function,otherwise%2C%20it%20will%20output%20zero.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ReLUActivationFunction(double input)
        {
            return (input > 0) ? input : 0;
        }

        /// <summary>
        /// Derivative (for back-propagation of ReLU).
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double DerivativeOfReLUActivationFunction(double input)
        {
            return (input > 0) ? 1: 0;
        }

        /// <summary>
        /// Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has 
        /// a small slope for negative values instead of a flat slope. The slope coefficient is determined before 
        /// training, i.e. it is not learnt during training. This type of activation function is popular in tasks 
        /// where we we may suffer from sparse gradients, for example training generative adversarial networks.
        /// 
        /// See: https://paperswithcode.com/method/leaky-relu#:~:text=Leaky%20Rectified%20Linear%20Unit%2C%20or,is%20not%20learnt%20during%20training.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static double LeakyReLUActivationFunction(double input)
        {
            return Math.Max(alpha * input, input);
        }

        /// <summary>
        /// Derivative (for back-propagation of LeakyReLU).
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double DerivativeOfLeakyReLUActivationFunction(double value)
        {
            return (value > 0) ? 1 : alpha; // return 1 if z > 0 else alpha
        }
        #endregion

        /// <summary>
        /// Create empty storage array for the neurons in the network.
        /// </summary>
        private void InitialiseNeurons()
        {
            List<double[]> neuronsList = new();

            // if layerDefinition is [2,3,2] ..   float[]
            // Neurons :      (o) (o)    <-2  ... [ 0, 0 ]
            //              (o) (o) (o)  <-3  ... [ 0, 0, 0 ]
            //                (o) (o)    <-2  ... [ 0, 0 ]
            //

            for (int layer = 0; layer < Layers.Length; layer++)
            {
                neuronsList.Add(new double[Layers[layer]]);
            }

            Neurons = neuronsList.ToArray();
        }

        /// <summary>
        /// Generate a cryptographic random number between -0.5...+0.5.
        /// </summary>
        /// <returns></returns>
        private static float RandomFloatBetweenMinusHalfToPlusHalf()
        {
            return (float)(RandomNumberGenerator.GetInt32(0, 10000) - 5000) / 10000;
        }

        /// <summary>
        /// initializes and populates biases.
        /// </summary>
        private void InitialiseBiases()
        {
            List<double[]> biasList = new();

            // for each layer of neurons, we have to set biases.
            for (int layer = 1; layer < Layers.Length; layer++)
            {
                double[] bias = new double[Layers[layer]];

                for (int biasLayer = 0; biasLayer < Layers[layer]; biasLayer++)
                {
                    bias[biasLayer] = RandomFloatBetweenMinusHalfToPlusHalf();
                }

                biasList.Add(bias);
            }

            Biases = biasList.ToArray();
        }

        /// <summary>
        /// initializes random array for the weights being held in the network.
        /// </summary>
        private void InitialiseWeights()
        {
            List<double[][]> weightsList = new(); // used to construct weights, as dynamic arrays aren't supported

            for (int layer = 1; layer < Layers.Length; layer++)
            {
                List<double[]> layerWeightsList = new();

                int neuronsInPreviousLayer = Layers[layer - 1];

                for (int neuronIndexInLayer = 0; neuronIndexInLayer < Neurons[layer].Length; neuronIndexInLayer++)
                {
                    double[] neuronWeights = new double[neuronsInPreviousLayer];

                    for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < neuronsInPreviousLayer; neuronIndexInPreviousLayer++)
                    {
                        neuronWeights[neuronIndexInPreviousLayer] = RandomFloatBetweenMinusHalfToPlusHalf();
                    }

                    layerWeightsList.Add(neuronWeights);
                }

                weightsList.Add(layerWeightsList.ToArray());
            }

            Weights = weightsList.ToArray();
        }

        /// <summary>
        /// Feed forward, inputs >==> outputs.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        internal double[] FeedForward(double[] inputs)
        {
            // put the INPUT values into layer 0 neurons
            for (int i = 0; i < inputs.Length; i++)
            {
                Neurons[0][i] = inputs[i];
            }

            // we start on layer 1 as we are computing values from prior layers (layer 0 is inputs)

            for (int layer = 1; layer < Layers.Length; layer++)
            {
                for (int neuronIndexForLayer = 0; neuronIndexForLayer < Layers[layer]; neuronIndexForLayer++)
                {
                    // sum of outputs from the previous layer
                    double value = 0f;

                    for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < Layers[layer - 1]; neuronIndexInPreviousLayer++)
                    {
                        // remember: the "weight" amplifies or reduces, so we take the output of the prior neuron and "amplify/reduce" it's output here
                        value += Weights[layer - 1][neuronIndexForLayer][neuronIndexInPreviousLayer] * Neurons[layer - 1][neuronIndexInPreviousLayer];
                    }

                    // any neuron fires or not based on the input. The point of a bias is to move the activation up or down.
                    // e.g. the value could be 0.3, adding a bias of 0.5 takes it to 0.8. You might think why not just use the weights to achieve this
                    // but remember weights are individual per prior layer neurons, the bias affects the SUM() of them.

                    Neurons[layer][neuronIndexForLayer] = activationMethod[layer](value + Biases[layer - 1][neuronIndexForLayer]);
                }
            }
            
            return Neurons[^1]; // final* layer contains OUTPUT
        }

        /// <summary>
        /// Back propagation
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="expected"></param>
        public void BackPropagate(double[] inputs, double[] expected)
        {
            // we need to ensure neurons have the right value otherwise gamma will be wrong.

            double[] output = FeedForward(inputs);
            double[][] gamma;

            List<double[]> gammaList = new();
            
            for (int i = 0; i < Layers.Length; i++)
            {
                gammaList.Add(new double[Layers[i]]);
            }

            gamma = gammaList.ToArray(); // gamma initialization

            int layer = Layers.Length - 2;

            for (int i = 0; i < output.Length; i++)
            {
                gamma[Layers.Length - 1][i] = (output[i] - expected[i]) * activationDerivativeMethod[layer](output[i]);             
            }

            for (int i = 0; i < Layers[^1]; i++) // calculates the w' and b' for the last layer in the network
            {
                Biases[Layers.Length - 2][i] -= gamma[Layers.Length - 1][i] * learningRate;

                for (int j = 0; j < Layers[^2]; j++)
                {
                    Weights[Layers.Length - 2][i][j] -= gamma[Layers.Length - 1][i] * Neurons[Layers.Length - 2][j] * learningRate;                 
                }
            }

            for (int i = Layers.Length - 2; i > 0; i--) // runs on all hidden layers
            {
                layer = i - 1;

                for (int j = 0; j < Layers[i]; j++) // outputs
                {
                    gamma[i][j] = 0;

                    for (int k = 0; k < gamma[i + 1].Length; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * Weights[i][k][j];
                    }

                    gamma[i][j] *= activationDerivativeMethod[layer](Neurons[i][j]); //calculate gamma
                }

                for (int j = 0; j < Layers[i]; j++) // iterate over outputs of layer
                {
                    Biases[i - 1][j] -= gamma[i][j] * learningRate; // modify biases of network

                    for (int k = 0; k < Layers[i - 1]; k++) // iterate over inputs to layer
                    {
                        Weights[i - 1][j][k] -= gamma[i][j] * Neurons[i - 1][k] * learningRate; // modify weights of network
                    }
                }
            }
        }
  
        /// <summary>
        /// Outputs the c# code for the network.
        /// </summary>
        /// <returns></returns>
        internal string Formula(ActivationFunctions[] func)
        {
            Dictionary<string, string> values = new();

            int neurons = 0;
            for (int layer = 0; layer < Layers.Length; layer++)
            {
                neurons += Neurons[layer].Length;
                if (neurons > 30) return "too big to output (exceed 30 neurons)";
            }
           
            for (int neuronIndexForLayer = 0; neuronIndexForLayer < Neurons[0].Length; neuronIndexForLayer++)
            {
                values.Add($"0-{neuronIndexForLayer}", $"input[{neuronIndexForLayer}]");
            }

            // we start on layer 1 as we are computing values from prior layers (layer 0 is inputs)
            for (int layer = 1; layer < Layers.Length; layer++)
            {
                List<string> dictionaryEntriesWeShouldRemove = new();

                for (int neuronIndexForLayer = 0; neuronIndexForLayer < Neurons[layer].Length; neuronIndexForLayer++)
                {
                    // sum of outputs from the previous layer
                    StringBuilder valueFormula = new(20);
                    valueFormula.AppendLine("");

                    for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < Neurons[layer - 1].Length; neuronIndexInPreviousLayer++)
                    {
                        // remember: the "weight" amplifies or reduces, so we take the output of the prior neuron and "amplify/reduce" it's output here
                        string weight = Weights[layer - 1][neuronIndexForLayer][neuronIndexInPreviousLayer].ToString();

                        string key = $"{layer - 1}-{neuronIndexInPreviousLayer}";
                        string neuronvalue = values[key];

                        if (!dictionaryEntriesWeShouldRemove.Contains(key)) dictionaryEntriesWeShouldRemove.Add(key);
                        // value += Weights[layer - 1][neuronIndexForLayer][neuronIndexInPreviousLayer] * Neurons[layer-1][neuronIndexInPreviousLayer];
                        //          ^ weight                                                                             ^ neuronValue
                        valueFormula.AppendLine($"(/* weight L{layer - 1}.N{neuronIndexForLayer}-L{layer - 1}.N{neuronIndexInPreviousLayer} x value */ {weight}*{neuronvalue})+");
                    }

                    valueFormula.AppendLine("");
                    string value = valueFormula.ToString().Trim('+');

                    // Neurons[layer][neuronIndexForLayer] = Activate(value + Biases[layer][neuronIndexForLayer]);
                    switch (func[layer])
                    {
                        case ActivationFunctions.ReLU:
                            values.Add($"{layer}-{neuronIndexForLayer}", $"/* L{layer}.N{neuronIndexForLayer} -> */ ({value}+{Biases[layer - 1][neuronIndexForLayer]})>0 ? 1 : 0");
                            break;

                        case ActivationFunctions.LeakyReLU:
                            values.Add($"{layer}-{neuronIndexForLayer}", $"/* L{layer}.N{neuronIndexForLayer} -> */ Math.Max({alpha}*{value}+{Biases[layer - 1][neuronIndexForLayer]}, {value}+{Biases[layer - 1][neuronIndexForLayer]})");
                            break;

                        case ActivationFunctions.Sigmoid:
                            values.Add($"{layer}-{neuronIndexForLayer}", $"/* L{layer}.N{neuronIndexForLayer} -> */ Math.Exp({value}+{Biases[layer - 1][neuronIndexForLayer]}) / (1.0f + Math.Exp({value}+{Biases[layer - 1][neuronIndexForLayer]}))");
                            break;

                        case ActivationFunctions.TanH:
                            values.Add($"{layer}-{neuronIndexForLayer}", $"/* L{layer}.N{neuronIndexForLayer} -> */ Math.Tanh({value}+{Biases[layer - 1][neuronIndexForLayer]})");
                            break;
                    }
                }

                // reduce dictionary, as each iteration embeds the previous layer
                foreach (string key in dictionaryEntriesWeShouldRemove) values.Remove(key);
            }

            string result = ($"        double[] outputFromNeuralNetwork = new double[{(Neurons[^1].Length > 1 ? 2 : 1)}];\n" +
                             $"        outputFromNeuralNetwork[0] = {values[(Layers.Length - 1).ToString() + "-0"]};\n\n").Replace("+-", "-");

            File.WriteAllText(@"c:\temp\formula.txt",result);
            return result;
        }
    }
}