//#define using_LeakuRELU
using AI;

Console.ForegroundColor = ConsoleColor.Yellow;

Console.WriteLine(@"                         _   ___       ___                                        _   ");
Console.WriteLine(@" /\   /\_____      _____| | /___\_ __ / __\___  _ __  ___  ___  _ __   __ _ _ __ | |_ ");
Console.WriteLine(@" \ \ / / _ \ \ /\ / / _ \ |//  // '__/ /  / _ \| '_ \/ __|/ _ \| '_ \ / _` | '_ \| __|");
Console.WriteLine(@"  \ V / (_) \ V  V /  __/ / \_//| | / /__| (_) | | | \__ \ (_) | | | | (_| | | | | |_ ");
Console.WriteLine(@"   \_/ \___/ \_/\_/ \___|_\___/ |_| \____/\___/|_| |_|___/\___/|_| |_|\__,_|_| |_|\__|");
Console.WriteLine("");

Console.ForegroundColor = ConsoleColor.White;

int[] layers = new int[2] { 26, 1 };

ActivationFunctions[] activationFunctions = new ActivationFunctions[2] { ActivationFunctions.TanH, ActivationFunctions.TanH }; // Tanh. Sigmoid also works, but ReLU can be temperamental even after 5000000 generations.

NeuralNetwork networkToTeachVowels = new(layers, activationFunctions);

bool outputTrainingData = true;

Console.WriteLine("TRAINING DATA");

float accuracyExpectedBetterThanDiffOf = 0.05F;

bool trained = false;
for (int generations = 0; generations < 5000000; generations++) // 4000 tanh, leakyRelu | 37,000 for Sigmoid based on accuracyExpectedBetterThanDiffOf = 0.05F; > 990000 leakyReLU if 0.001F accuracy.
{
    // make 26 inputs 0..25, one per letter
    // for each letter to train we put a "1" in the array, and if it's a vowel teach it to return "1"
    for (int letter = 0; letter < 26; letter++)
    {
        double[] letters = new double[26]; // initialise all zero
        letters[letter] = 1; // set the letter we are processing to "1"

        double desiredOutput1ifVowel0IfConsonant = (letter == 0 || letter == 4 || letter == 8 || letter == 14 || letter == 20) ? 1 : 0;

        networkToTeachVowels.BackPropagate(letters, new double[] { desiredOutput1ifVowel0IfConsonant });

        if (outputTrainingData)
        {
            Console.WriteLine(char.ConvertFromUtf32(letter + 65) + " " + String.Join(",", letters) + " => " + desiredOutput1ifVowel0IfConsonant);
        }
    }

    trained = true;
    for (int letter = 0; letter < 26; letter++)
    {
        double[] letters = new double[26]; // initialise all zero
        letters[letter] = 1; // set the letter we are processing to "1"

        double desiredOutput1ifVowel0IfConsonant = (letter == 0 || letter == 4 || letter == 8 || letter == 14 || letter == 20) ? 1 : 0;

        if (Math.Abs(networkToTeachVowels.FeedForward(letters)[0] - desiredOutput1ifVowel0IfConsonant) > accuracyExpectedBetterThanDiffOf)
        {
            trained = false;
            break;
        }
    }

    if (trained) break;

    if (generations % 1000 == 0) Console.WriteLine($"Epoch {generations}");
    outputTrainingData = false;
}

if (!trained)
{
    Console.BackgroundColor= ConsoleColor.Red;
    Console.WriteLine("!! NOT SUCCESSFULLY TRAINED !!");
    Console.BackgroundColor = ConsoleColor.Black;
}
else
{
    Console.BackgroundColor = ConsoleColor.Green;
    Console.WriteLine("SUCCESSFULLY TRAINED");
    Console.BackgroundColor = ConsoleColor.Black;

}

Console.WriteLine("");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine("TEST OF TRAINED NETWORK:");
Console.ForegroundColor = ConsoleColor.White;

// show the result of training
for (int letter = 0; letter < 26; letter++)
{
    double[] letters = new double[26];
    letters[letter] = 1;

    double desiredOutput1ifVowel0IfConsonant = (letter == 0 || letter == 4 || letter == 8 || letter == 14 || letter == 20) ? 1 : 0;

    Console.WriteLine($"{char.ConvertFromUtf32(65 + letter)} {(networkToTeachVowels.FeedForward(letters)[0] > 0.5 ? "Vowel" : "Consonant")}. Difference: {(networkToTeachVowels.FeedForward(letters)[0] - desiredOutput1ifVowel0IfConsonant):0.###}");
}

Console.WriteLine("");
Console.ForegroundColor = ConsoleColor.Blue;
Console.WriteLine("NEURAL NETWORK AS MATHEMATICAL REPRESENTATION");
Console.ForegroundColor = ConsoleColor.White;
Console.WriteLine(networkToTeachVowels.Formula(activationFunctions));
Console.WriteLine("");
Console.WriteLine("Using the hard-coded formula:");
for (int letter = 0; letter < 26; letter++)
{
    double[] input = new double[26];
    input[letter] = 1;

    // Below formula was derived from the training ^ Formula(). It may slightly different in weightings/biases each time (due to starting values).
    double[] outputFromNeuralNetwork = new double[1];

#if using_LeakuRELU
    outputFromNeuralNetwork[0] = /* L1.N0 -> */ Math.Max(0.01 *
    (/* weight L0.N0-L0.N0 x value */ 0.8521055873616917 * input[0]) +
    (/* weight L0.N0-L0.N1 x value */ -0.14789439509698823 * input[1]) +
    (/* weight L0.N0-L0.N2 x value */ -0.147894377555643 * input[2]) +
    (/* weight L0.N0-L0.N3 x value */ -0.2392824999461301 * input[3]) +
    (/* weight L0.N0-L0.N4 x value */ 0.8521055485884536 * input[4]) +
    (/* weight L0.N0-L0.N5 x value */ -0.14789443387026033 * input[5]) +
    (/* weight L0.N0-L0.N6 x value */ -0.18143839175854723 * input[6]) +
    (/* weight L0.N0-L0.N7 x value */ -0.1478944498762773 * input[7]) +
    (/* weight L0.N0-L0.N8 x value */ 0.8521055676650077 * input[8]) +
    (/* weight L0.N0-L0.N9 x value */ -0.1494348070807166 * input[9]) +
    (/* weight L0.N0-L0.N10 x value */ -0.14789441633422892 * input[10]) +
    (/* weight L0.N0-L0.N11 x value */ -0.14789439879291336 * input[11]) +
    (/* weight L0.N0-L0.N12 x value */ -0.14789438125157292 * input[12]) +
    (/* weight L0.N0-L0.N13 x value */ -0.1605724699738638 * input[13]) +
    (/* weight L0.N0-L0.N14 x value */ 0.8521056236104192 * input[14]) +
    (/* weight L0.N0-L0.N15 x value */ -0.14789435884822089 * input[15]) +
    (/* weight L0.N0-L0.N16 x value */ -0.2478961102759339 * input[16]) +
    (/* weight L0.N0-L0.N17 x value */ -0.14789444131859927 * input[17]) +
    (/* weight L0.N0-L0.N18 x value */ -0.14789442377730705 * input[18]) +
    (/* weight L0.N0-L0.N19 x value */ -0.14789440623598957 * input[19]) +
    (/* weight L0.N0-L0.N20 x value */ 0.852105611305353 * input[20]) +
    (/* weight L0.N0-L0.N21 x value */ -0.15273875088586203 * input[21]) +
    (/* weight L0.N0-L0.N22 x value */ -0.19239081176033015 * input[22]) +
    (/* weight L0.N0-L0.N23 x value */ -0.1478944204990266 * input[23]) +
    (/* weight L0.N0-L0.N24 x value */ -0.19265061700803696 * input[24]) +
    (/* weight L0.N0-L0.N25 x value */ -0.14789444771840612 * input[25]) +

    +0.14789616652110407,
    (/* weight L0.N0-L0.N0 x value */ 0.8521055873616917 * input[0]) +
    (/* weight L0.N0-L0.N1 x value */ -0.14789439509698823 * input[1]) +
    (/* weight L0.N0-L0.N2 x value */ -0.147894377555643 * input[2]) +
    (/* weight L0.N0-L0.N3 x value */ -0.2392824999461301 * input[3]) +
    (/* weight L0.N0-L0.N4 x value */ 0.8521055485884536 * input[4]) +
    (/* weight L0.N0-L0.N5 x value */ -0.14789443387026033 * input[5]) +
    (/* weight L0.N0-L0.N6 x value */ -0.18143839175854723 * input[6]) +
    (/* weight L0.N0-L0.N7 x value */ -0.1478944498762773 * input[7]) +
    (/* weight L0.N0-L0.N8 x value */ 0.8521055676650077 * input[8]) +
    (/* weight L0.N0-L0.N9 x value */ -0.1494348070807166 * input[9]) +
    (/* weight L0.N0-L0.N10 x value */ -0.14789441633422892 * input[10]) +
    (/* weight L0.N0-L0.N11 x value */ -0.14789439879291336 * input[11]) +
    (/* weight L0.N0-L0.N12 x value */ -0.14789438125157292 * input[12]) +
    (/* weight L0.N0-L0.N13 x value */ -0.1605724699738638 * input[13]) +
    (/* weight L0.N0-L0.N14 x value */ 0.8521056236104192 * input[14]) +
    (/* weight L0.N0-L0.N15 x value */ -0.14789435884822089 * input[15]) +
    (/* weight L0.N0-L0.N16 x value */ -0.2478961102759339 * input[16]) +
    (/* weight L0.N0-L0.N17 x value */ -0.14789444131859927 * input[17]) +
    (/* weight L0.N0-L0.N18 x value */ -0.14789442377730705 * input[18]) +
    (/* weight L0.N0-L0.N19 x value */ -0.14789440623598957 * input[19]) +
    (/* weight L0.N0-L0.N20 x value */ 0.852105611305353 * input[20]) +
    (/* weight L0.N0-L0.N21 x value */ -0.15273875088586203 * input[21]) +
    (/* weight L0.N0-L0.N22 x value */ -0.19239081176033015 * input[22]) +
    (/* weight L0.N0-L0.N23 x value */ -0.1478944204990266 * input[23]) +
    (/* weight L0.N0-L0.N24 x value */ -0.19265061700803696 * input[24]) +
    (/* weight L0.N0-L0.N25 x value */ -0.14789444771840612 * input[25]) +

    +0.14789616652110407);
#else // tanh

    outputFromNeuralNetwork[0] = /* L1.N0 -> */ Math.Tanh(
        (/* weight L0.N0-L0.N0 x value */ 2.473264012333785 * input[0]) +
        (/* weight L0.N0-L0.N1 x value */ -0.5701099631486187 * input[1]) +
        (/* weight L0.N0-L0.N2 x value */ -0.5701098698734448 * input[2]) +
        (/* weight L0.N0-L0.N3 x value */ -0.5701097765980829 * input[3]) +
        (/* weight L0.N0-L0.N4 x value */ 2.4734041700284215 * input[4]) +
        (/* weight L0.N0-L0.N5 x value */ -0.570110093601201 * input[5]) +
        (/* weight L0.N0-L0.N6 x value */ -0.5701100003262883 * input[6]) +
        (/* weight L0.N0-L0.N7 x value */ -0.5701099070511886 * input[7]) +
        (/* weight L0.N0-L0.N8 x value */ 2.473431813211454 * input[8]) +
        (/* weight L0.N0-L0.N9 x value */ -0.570110224009139 * input[9]) +
        (/* weight L0.N0-L0.N10 x value */ -0.5701101307344887 * input[10]) +
        (/* weight L0.N0-L0.N11 x value */ -0.5701100374596495 * input[11]) +
        (/* weight L0.N0-L0.N12 x value */ -0.5701099441846242 * input[12]) +
        (/* weight L0.N0-L0.N13 x value */ -0.5701098509094118 * input[13]) +
        (/* weight L0.N0-L0.N14 x value */ 2.473223134865146 * input[14]) +
        (/* weight L0.N0-L0.N15 x value */ -0.5701101682088007 * input[15]) +
        (/* weight L0.N0-L0.N16 x value */ -0.5701100749340386 * input[16]) +
        (/* weight L0.N0-L0.N17 x value */ -0.5701099816590883 * input[17]) +
        (/* weight L0.N0-L0.N18 x value */ -0.5701098883839506 * input[18]) +
        (/* weight L0.N0-L0.N19 x value */ -0.5701097951086265 * input[19]) +
        (/* weight L0.N0-L0.N20 x value */ 2.47330505052345 * input[20]) +
        (/* weight L0.N0-L0.N21 x value */ -0.570110112273923 * input[21]) +
        (/* weight L0.N0-L0.N22 x value */ -0.570110018999048 * input[22]) +
        (/* weight L0.N0-L0.N23 x value */ -0.5701099257239856 * input[23]) +
        (/* weight L0.N0-L0.N24 x value */ -0.5701098324487359 * input[24]) +
        (/* weight L0.N0-L0.N25 x value */ -0.570109739173299 * input[25]) +
        +0.57011887836769);
#endif

    Console.WriteLine(char.ConvertFromUtf32(65 + letter) + " " + (outputFromNeuralNetwork[0] > 0.5 ? "Vowel" : "Consonant"));
}

Console.WriteLine();
Console.WriteLine("How does the above work? (the values reference the code above lines 113-200)");
Console.WriteLine("Most of the neurons (i.e. consonants) do nothing; the biases/weights will return 0, because the input will be zero");
Console.WriteLine("In fact, if you notice all consonants are approx. -0.5701, which when added to +0.5701 cancel out to return 0 (tanh 0=0)");
Console.WriteLine("For vowels, 2.4733 gets added to 0.57012 => (tanh 3.04342 = 0.996) or near enough 1.");

// Classification really is simple. Back propagation has led it to associate 1 with a,e,i,o,u and 0 with the others.

Console.ReadLine();