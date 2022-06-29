(* ::Package:: *)

(* 
  TODO:
  - Support loss function on multiple bit output ports
  - bias term
*)

(* ------------------------------------------------------------------ *)
BeginPackage["neurallogic`"]
(* ------------------------------------------------------------------ *)

BinaryNN::usage = "Specifies a binary neural network and returns a {soft, hard} binary neural network pair.";
AppendLoss::usage = "Appends a loss function to a (soft) binary neural network.";
Harden::usage = "Convert soft-bit to hard-bit.";
Soften::usage = "Convert hard-bit to soft-bit.";
HardBinaryNN::usage = "Constructs a hard binary neural network from a trained soft binary neural network.";
ExtractWeights::usage = "Extract the weights from a soft neural network.";
NeuralMajority::usage = "Specifies a differentiable boolean majority function.";
WeightedNeuralMajority::usage = "Specifies a differentiable weighted boolean majority function.";
NeuralMajorityLayer::usage = "Specifies a neural majority layer.";
BooleanMajorityLayer::usage = "Specifies a boolean majority layer.";
BooleanMajorityChain::usage = "Specifies a boolean neural network chain.";
BitLoss::usage = "Specifies a bit loss function.";
NeuralMajorityForward::usage = "Forward pass of a neural majority layer.";
NeuralMajorityBackward::usage = "Backward pass of a neural majority layer.";
BitLossForward::usage = "Forward pass of a bit loss function.";
BitLossBackward::usage = "Backward pass of a bit loss function.";
RandomSoftBit::usage = "Randomly generate a soft bit.";
NeuralAND::usage = "Specifies a differentiable boolean AND function.";
NeuralANDLayer::usage = "Specifies a neural AND layer.";
NeuralOR::usage = "Specifies a differentiable boolean OR function.";
NeuralORLayer::usage = "Specifies a neural OR layer.";
NeuralNOTLayer::usage = "Specifies a neural NOT layer.";
NeuralDNFLayer::usage = "Specifies a neural DNF layer.";

(* ------------------------------------------------------------------ *)
Begin["`Private`"]
(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)
(* Boolean utilities *)
(* ------------------------------------------------------------------ *)

(* Soft-bits between -1 and +1 *)
Harden[softBit_] := If[softBit > 0.0, True, False]
Harden[softBit_List] := Map[Harden, softBit]

(* Hard-bits between 0 and 1 *)
Soften[hardBit_] := If[hardBit == 1, 1.0, -1.0]

RandomSoftBit[n_] := RandomReal[{-0.2, 0.2}, n]

(* ------------------------------------------------------------------ *)
(* Neural NOT layer *)
(* ------------------------------------------------------------------ *)

NeuralNOTLayer[weights_List] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Array" -> weights],
    "Not" -> FunctionLayer[
      MapThread[#1 #2 &, {#Input, #Weights}] &, 
      "Input" -> Length[weights]
    ]
    (*"NonLin" -> ElementwiseLayer["HardTanh"]*)
  |>,
  {
    "Weights" -> NetPort["Not", "Weights"]
    (*"Not" -> "NonLin"*)
  }
]

NeuralNOTLayer[n_] := NeuralNOTLayer[RandomSoftBit[n]]

(* ------------------------------------------------------------------ *)
(* Neural logic layer *)
(* ------------------------------------------------------------------ *)

NeuralLogicLayer[inputSize_, layerSize_, f_Function] := NetGraph[
  Association[Join[
      Map[
        "n" <> ToString[#] -> f[inputSize] &, 
        Range[layerSize]
      ],
      {
        "Catenate" -> CatenateLayer[]
        (*"NonLin" -> ElementwiseLayer["HardTanh"]*)
      }
  ]],
  Join[
    Map[
      "n" <> ToString[#] -> "Catenate" &,
      Range[layerSize]
    ]
    (*{"Catenate" -> "NonLin"}*)
  ]
]

(* ------------------------------------------------------------------ *)
(* Differentiable AND *)
(* ------------------------------------------------------------------ *)

(*
  Computes the soft-AND of all input soft-bits.
  The input soft-bits are assumed to be between -1 and +1.
  The output is a soft-bit between -1 and +1.

  The weight soft-bit controls how "active" the AND operation is.

  E.g. If the weight is -1 (i.e. fully false) then the AND operation
  is fully inoperative and the output is 1 (fully true) regardless
  of the input. In this case soft-AND acts like a nop.

  E.g. If the weight is +1 then the AND operation is fully operative
  and the output is the input bit. In this case soft-AND acts like a
  pass-through function.

  Each input soft-bit has its own associated weight, and therefore
  soft-participates in the overall output of the neuron.
*)
NeuralAND[weights_List] := NetGraph[
  <|
    "SoftInclude" -> FunctionLayer[
      MapThread[
        1 - 1/2 (1 + 1/2 (-1 - #1)) (1 + #2) &, 
        {#Input, #Weights}
      ] &, 
      "Input" -> Length[weights]
    ],
    "And1" -> FunctionLayer[Times],
    "And2" -> FunctionLayer[2.0 # - 1.0 &],
    "Weights" -> NetArrayLayer["Array" -> weights],
    "Reshape" -> ReshapeLayer[{1}]
  |>,
  {
   "Weights" -> NetPort["SoftInclude", "Weights"],
   "SoftInclude" -> "And1",
   "And1" -> "And2",
   "And2" -> "Reshape"
  }
]

NeuralAND[n_] := NeuralAND[RandomReal[{-0.001, 0.001}, n]]

NeuralANDLayer[inputSize_, layerSize_] := NeuralLogicLayer[inputSize, layerSize, NeuralAND[#] &]

(* ------------------------------------------------------------------ *)
(* Differentiable OR *)
(* ------------------------------------------------------------------ *)

NeuralOR[weights_List] := NetGraph[
  <|
    "SoftInclude" -> FunctionLayer[
      MapThread[1 - 1/4 (1 + #1) (1 + #2) &, {#Input, #Weights}] &, 
      "Input" -> Length[weights]
   ],
   "Or1" -> FunctionLayer[Times[#] &],
   "Or2" -> FunctionLayer[-1 + 2(1 - #) &],
   "Weights" -> NetArrayLayer["Array" -> weights],
   "Reshape" -> ReshapeLayer[{1}]
  |>,
  {
    "Weights" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "Or1",
    "Or1" -> "Or2",
    "Or2" -> "Reshape"
  }
]

NeuralOR[n_] := NeuralOR[RandomReal[{-0.001, 0.001}, n]]

NeuralORLayer[inputSize_, layerSize_] := NeuralLogicLayer[inputSize, layerSize, NeuralOR[#] &]

(* ------------------------------------------------------------------ *)
(* Differentiable DNF layer *)
(* ------------------------------------------------------------------ *)

(*
  TODO: work out the policy for the sizes that ensure all possible DNF expressions
  can be learned.
  TODO: then change parameter to [0, 1] from 0 capacity to full capacity to represent
  all possible DNF expressions.
  Ignoring NOTs then andSize does not need to be larger than 2^(inputSize - 1)
*)
NeuralDNFLayer[inputSize_, andSize_, orSize_] := NetGraph[
  <|
    "NOT layer" -> NeuralNOTLayer[inputSize],
    "AND layer" -> NeuralANDLayer[inputSize, andSize],
    "OR layer" -> NeuralORLayer[andSize, orSize]
  |>,
  {
    "NOT layer" -> "AND layer",
    "AND layer" -> "OR layer"
  }
]

(* ------------------------------------------------------------------ *)
(* Differentiable boolean majority *)
(* ------------------------------------------------------------------ *)

(*
  Computes the soft-majority of all input soft-bits.
  The input soft-bits are assumed to be between -1 and +1.
  The output is a soft-bit between -1 and +1.

  TODO.
*)
NeuralMajorityForward[] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]]
  },
  Block[
    {
      (* Currently using sort (probably compiles to QuickSort)
        - average O(n log n)
        - worst-case O(n^2)
      *)
      (* Replace with Floyd-Rivest algorithm:
        - average n + \min(k, n - k) + O(\sqrt{n \log n}) comparisons with probability at least 1 - 2n^{-1/2}
        - worst-case O(n^2)
        See https://danlark.org/2020/11/11/miniselect-practical-and-generic-selection-algorithms/
      *)
      s = Sort[MapIndexed[{#1, #2[[1]]*1.0} &, input]], 
      i = Floor[Ceiling[Length[input]/2.0]]
    },
    {First[s[[i]]], Last[s[[i]]]}
  ]
]
  
NeuralMajorityBackward[] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]],
    Typed[outgrad, TypeSpecifier["PackedArray"]["MachineReal", 1]],
    Typed[output, TypeSpecifier["PackedArray"]["MachineReal", 1]]
  },
  Block[
    {
      i = Round[Last[output]],
      g = First[outgrad]
    },
    Table[
      If[i == j, g, If[i < j, 0.0, 0.0]],
      {j, 1, Length[input]}
    ]
  ]
]

NeuralMajority[] := CompiledLayer[NeuralMajorityForward[], NeuralMajorityBackward[]]

(* ------------------------------------------------------------------ *)
(* Differentiable weighted boolean majority *)
(* ------------------------------------------------------------------ *)

WeightedNeuralMajority[weights_List] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Array" -> weights, "Output" -> Length[weights]],
    "WeightedBits" -> FunctionLayer[Clip[#Weights #Bits, {-1, 1}] &, "Output" -> Length[weights]],
    "Majority" -> NeuralMajority[],
    "MajorityBit" -> PartLayer[1],
    "Reshape" -> ReshapeLayer[{1}]
  |>,
  {
    "Weights" -> "WeightedBits",
    NetPort["Input"] -> "WeightedBits",
    "WeightedBits" -> "Majority",
    "Majority" -> "MajorityBit",
    "MajorityBit" -> "Reshape"
  }
]

WeightedNeuralMajority[n_] := WeightedNeuralMajority[RandomSoftBit[n]]

(* ------------------------------------------------------------------ *)
(* Majority layer *)
(* ------------------------------------------------------------------ *)

NeuralMajorityLayer[inputSize_, layerSize_] := NetGraph[
    Association[
      Map[
        "majority" <> ToString[#] -> WeightedNeuralMajority[inputSize] &, 
        Range[layerSize]
      ],
      "catenate" -> CatenateLayer[]
    ],
    Map[
      "majority" <> ToString[#] -> "catenate" &,
      Range[layerSize]
    ]
  ]

BooleanMajorityLayer[input_, weights_] := Map[Inner[Xnor, input, #, Majority] &, weights]


BooleanMajorityChain[spec_List] := Function[{input, weights},
  Block[{activations = input},
    Scan[
      Block[{layerWeights = #},
        activations = BooleanMajorityLayer[activations, layerWeights];
      ] &,
      weights
    ];
    activations
  ]
] 

(* ------------------------------------------------------------------ *)
(* Specify binary neural networks *)
(* ------------------------------------------------------------------ *)

SetAttributes[BinaryNN, HoldFirst]

BinaryNN[spec_List] := Module[{softNet, hardNet},
  softNet = NetChain[# & /@ spec];
  (* TODO: update for mixed layers *)
  hardNet = BooleanMajorityChain[spec];
  {softNet, hardNet}
]

(* ------------------------------------------------------------------ *)
(* Bit loss *)
(* ------------------------------------------------------------------ *)

(* First half of input is the prediction, and the second half of the input is the target *)
BitLossForward[inputSize_, eps_] := Function[
  {Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]]},
  Block[
    {
      predictedBits = Take[input, inputSize], 
      targetBits = Take[input, -inputSize]
    },
    Table[
      If[(predictedBits[[n]] > (0.0 + eps) && targetBits[[n]] > 0.0) || (predictedBits[[n]] < (0.0 - eps) && targetBits[[n]] < 0.0),
        0.0,
        (predictedBits[[n]] - targetBits[[n]])^2
      ], 
      {n, 1, inputSize}
    ]
  ]
]

BitLossBackward[inputSize_, eps_] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]],
    Typed[outgrad, TypeSpecifier["PackedArray"]["MachineReal", 1]]
  },
  Block[
    {
      predictedBits = Take[input, inputSize], 
      targetBits = Take[input, -inputSize]
    },
    Table[
      If[n <= inputSize,
        If[(predictedBits[[n]] > (0.0 + eps) && targetBits[[n]] > 0.0) || (predictedBits[[n]] < (0.0 - eps) && targetBits[[n]] < 0.0),
          0.0,
          2 (predictedBits[[n]] - targetBits[[n]])
        ],
        0.0
      ], 
      {n, 1, 2 inputSize}
    ]
  ]
]

(* 
  Small margin (e.g. 0.01) accelerates learning although trajectory is noisier. 
  Large margin (e.g. 1.0) slows learning but trajectory is more smooth.
*)
BitLoss[inputSize_] := With[{eps = 1.0}, 
  CompiledLayer[
    BitLossForward[inputSize, eps], 
    BitLossBackward[inputSize, eps], 
    "Input" -> {2 inputSize}, 
    "Output" -> {inputSize}
  ]
]

(* ------------------------------------------------------------------ *)
(* Trainable (soft) binary neural network *)
(* ------------------------------------------------------------------ *)

AppendLoss[net_] := Block[{netOutputSize = NetExtract[net, "Output"]},
  NetGraph[
    <|
      "NeuralLogicNet" -> net,
      "Catenate" -> CatenateLayer[],
      "BitLoss" -> BitLoss[netOutputSize],
      "loss" -> SummationLayer[]
    |>,
    {
      "NeuralLogicNet" -> "Catenate",
      NetPort["Target"] -> "Catenate",
      "Catenate" -> "BitLoss",
      "BitLoss" -> "loss",
      "loss" -> NetPort["Loss"]
    }
  ]
]

(* ------------------------------------------------------------------ *)
(* Network hardening *)
(* ------------------------------------------------------------------ *)

(* TODO: update for mixed layers *)
ExtractWeights[net_] := Module[{layers, layerSizes, majorityNames, majorityNeuronLayers},
  layers = NetExtract[net, All];
  layerSizes = Information[#, "ArraysCount"] & /@ layers;
  majorityNames = Map[Map[{"majority" <> ToString[#]} &, Range[#]] &, layerSizes];
  majorityNeuronLayers = MapThread[NetExtract, {layers, majorityNames}];
  Map[
    Map[Normal[NetExtract[NetExtract[#, "Weights"], "Arrays"]["Array"]] &, #] &,
    majorityNeuronLayers
  ]
]

(* TODO: update for mixed layers *)
HardBinaryNN[hardNet_, trainedSoftNet_] := Module[{hardenedWeights, inputSize},
  hardenedWeights = Harden[ExtractWeights[trainedSoftNet]];
  inputSize = Length[First[First[hardenedWeights]]];
  inputSymbols = Map[Symbol["x" <> ToString[#]] &, Range[inputSize]];
  Function[Evaluate[inputSymbols], Evaluate[hardNet[inputSymbols, hardenedWeights]]]
]

End[]

EndPackage[]
