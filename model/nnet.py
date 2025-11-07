# nnet.py
import torch
# import torch.nn.functional as F

# Feedforward MLP (input_layer -> hidden_layers -> output_layer)
class VanillaNet(torch.nn.Module):
    def __init__(self, input_units, output_units, deep_layers_units=[1000, 200, 50], dropout_rate=0.0, activation='relu'):
        super().__init__()
        if activation.lower() == "relu":
            act = torch.nn.ReLU
        elif activation.lower() in ["leakyrelu", "leaky_relu"]:
            act = torch.nn.LeakyReLU
        elif activation.lower() == "elu":
            act = torch.nn.ELU
        elif activation.lower() == "gelu":
            act = torch.nn.GELU
        elif activation.lower() == "silu":
            act = torch.nn.SiLU
        else:
            raise ValueError(f"Unsupported activation function {activation}")
        layers = [input_units]+deep_layers_units
        self.deep_layers = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            deep_layer = torch.nn.Sequential(
                torch.nn.Linear(layers[i], layers[i+1]),
                torch.nn.BatchNorm1d(layers[i+1], momentum=0.9),
                # torch.nn.LayerNorm(layers[i+1]),
                act(),
            )
            if dropout_rate is not None and dropout_rate > 0.0:
                deep_layer.append(torch.nn.Dropout(p=dropout_rate))
            self.deep_layers.append(deep_layer)
        self.output_layer = torch.nn.Linear(deep_layers_units[-1], output_units)
        
    def forward(self, x_in):
        x = x_in
        for layer in self.deep_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self(x)
        return outputs


# Ridge regression model
class LinearRidge(torch.nn.Module):
    def __init__(self, input_units, output_units=1):
        super().__init__()
        self.fc = torch.nn.Linear(input_units, output_units)
    def forward(self, x):
        return self.fc(x)

