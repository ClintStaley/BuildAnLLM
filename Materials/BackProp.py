import numpy as np
import numpy.random as rnd
import json, sys

class Layer:
    # dim  -- number of neurons in the layer
    # prev -- prior layer, or None if input layer.  If None, ignore all
    #  following params.
    # act -- activation function: accept np.array of z's, return np.array of a's
    # act_prime -- derivative function: accept np.arrays of z's and of a's,
    #  return derivative of activation wrt z's, 1-D or 2-D as appropriate
    # weights -- initial weights. Set to small random if "None".
    #
    # All the following member data are None for input layer
    # in_weights -- matrix of input weights
    # in_derivs -- derivatives of E/weight for last sample
    # zs -- z values for last sample
    # z_derivs -- derivatives of E/Z for last sample
    # batch_derivs -- cumulative sum of in_derivs across current batch
    def __init__(self, dim, prev, act, act_prime, weights=None):
        self.dim = dim
        self.prev = prev
        if prev is None:
            return
            
        self.prev.next = self
        self.act = act
        self.act_prime = act_prime

        # Glorot (Xavier) uniform initialization; bias column zero-initialized
        if weights is not None:
            self.in_weights = weights
        else:
            fan_in = prev.dim
            fan_out = dim
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.in_weights = np.empty((dim, prev.dim + 1), dtype=np.float32)
            self.in_weights[:, :prev.dim] = rnd.uniform(
                -limit, limit, size=(dim, prev.dim)
            ).astype(np.float32)
            self.in_weights[:, prev.dim] = 0.0
        self.in_derivs = np.zeros((dim, prev.dim+1), dtype=np.float32)
        self.zs = np.zeros((dim), dtype=np.float32)
        self.z_derivs = np.zeros((dim), dtype=np.float32)
        self.batch_derivs = np.zeros((dim, prev.dim+1), dtype=np.float32)
    
    def get_dim(self): return self.dim
    
    # Return the dE/dW for the weight from previous layer's |src| neuron to
    # our |trg| neuron. 
    def get_deriv(self, src, trg):
        return self.in_derivs[trg][src]
        
    # Compute self.outputs, using vals if given, else using outputs from
    # previous layer and passing through our in_weights and activation.
    def propagate(self, vals = None):
        if vals is not None:
            self.outputs = vals
        else:
            self.zs = self.in_weights.dot(np.append(self.prev.outputs, 1.0))
            self.outputs = self.act(self.zs);
    
    # Compute self.in_derivs, assuming 
    # 1. We have a prev layer (else in_derivs is None)
    # 2. Either
    #    a. There is a next layer with correct z_derivs, OR
    #    b. The provided err_prime function accepts np arrays 
    #       of outputs and of labels, and returns an np array 
    #       of dE/da for each output
    def backpropagate(self, err_prime=None, labels=None):
        act_partials = self.act_prime(self.zs, self.outputs)
        
        if err_prime:
            out_derivs = err_prime(self.outputs, labels)
        else:        
            out_derivs = \
             np.transpose(self.next.in_weights).dot(self.next.z_derivs)[:-1]
        
        if len(act_partials.shape) == 1:
            self.z_derivs = act_partials * out_derivs
        else:
            self.z_derivs = act_partials.dot(out_derivs)
        
        self.in_derivs = np.outer(self.z_derivs,
         np.append(self.prev.outputs, 1))
        self.batch_derivs += self.in_derivs

    # Adjust all weights by avg gradient accumulated for current batch * -|rate|
    def apply_batch(self, batch_size, rate):
        self.in_weights += -self.batch_derivs * (rate/batch_size)
        
    # Reset internal data for start of a new batch
    def start_batch(self):
        self.batch_derivs.fill(0.0)
    
    def tweak_weight(self, src, trg, delta):
        self.in_weights[trg][src] += delta
        
    # Return string description of self for debugging
    def __repr__(self):
        return "Outs: " + str(self.outputs) if self.prev is None else \
         "Wgts: {}\nWgt Derivs: {}\nZs: {} Z Derivs: {}\n Outs: {}". \
         format(self.in_weights, self.in_derivs, self.zs, self.z_derivs,
          self.outputs)

def cross_entropy_err(outs, lbls):
    entropy = 0.0
    cross_entropy = 0.0
    for lbl, out in zip(lbls, outs):
        if lbl > 0.0:
            cross_entropy -= lbl * np.log(out)
            entropy -= lbl * np.log(lbl)

    return cross_entropy - entropy

class Network:
    # arch -- list of (dim, act) pairs
    # err -- error function: "cross_entropy" or "mse"
    # wgts -- list of one 2-d np.array per layer in arch
    def __init__(self, arch, err, wgts = None):
        epsilon = sys.float_info.epsilon
        if err == 'cross_entropy':
            self.err = cross_entropy_err
            # Guard against division by zero for numerical stability
            self.err_prime = lambda outs, lbls: -lbls/np.clip(outs, epsilon, 1.0)
        else:
            print("Unknown loss {}".format(err))
            return
        
        self.layers = layers = []
        for dim, act_descr in arch:
            if act_descr == 'relu':
                act = lambda z : z * (z > 0)
                act_prime = lambda z, a : (z > 0).astype(np.float32)
            elif act_descr == 'softmax':
                # Numerically stable softmax
                act = lambda z : (
                    (lambda t: np.exp(t) / np.sum(np.exp(t)))(z - np.max(z))
                )
                act_prime = lambda z, a : np.diag(a) - np.outer(a,a)
            else:
                act = act_prime = None
            
            layers.append(Layer(dim, layers[-1] if len(layers) else None, act,
                act_prime, wgts[len(layers)-1] if wgts is not None else None))
    
    # Forward propagate, passing inputs to first layer, and returning outputs
    # of final layer
    def predict(self, inputs):
        self.layers[0].propagate(inputs)
        for layer in self.layers[1:]:
            layer.propagate()

        return self.layers[-1].outputs
        
    # Assuming forward propagation is done, return current error, assuming
    # expected final layer output is |labels|
    def get_err(self, labels):
        return self.err(self.layers[-1].outputs, labels)
    
    # Assuming a predict was just done, update all in_derivs, and add to batch_derivs
    def backpropagate(self, labels):
        print("\nDoing backprop step ----------")
        self.layers[-1].backpropagate(self.err_prime, labels)
        for i in range(len(self.layers)-2, 0, -1):
            self.layers[i].backpropagate()
        for i, layer in enumerate(self.layers):
            print("\nLayer {} ---------------------\n{}".format(i, repr(layer)))
    
    # Verify all partial derivatives for weights by adding an
    # epsilon value to each weight and rerunning prediction toZZ
    # see if change in error correctly reflects weight change
    def validate_derivs(self, inputs, outputs):
        eps = .01
        self.predict(inputs)
        base_line = self.get_err(outputs)
        self.backpropagate(outputs)
        for lyr in range(1, len(self.layers)):
            src_lyr = self.layers[lyr-1]
            trg_lyr = self.layers[lyr]
            for trg in range(trg_lyr.get_dim()):
                for src in range(src_lyr.get_dim() + 1):
                    trg_lyr.tweak_weight(src, trg, eps)
                    self.predict(inputs)
                    new_err = self.get_err(outputs)
                    change = new_err - base_line
                    expected_change = trg_lyr.get_deriv(src, trg)*eps
                    print(("Test {}/{} to {}/{}: {:.6f} - {:.6f}"
                     " = {:.6f} ({:.4f}% error)").format(lyr-1, src, lyr, trg,
                     new_err, base_line, change, change if change == 0.0 else
                     abs((expected_change-change) * 100 / expected_change)))
                     
                    trg_lyr.tweak_weight(src, trg, -eps)
    
    # Run a batch, assuming |data| holds input/output pairs comprising the batch
    # Forward propagate for each input, record error, and backpropagate.  At batch
    # end, report average error for the batch, and do a derivative update.
    def run_batch(self, data, rate):
        batch_size = len(data)
        for lyr in self.layers[1:]:
            lyr.start_batch()
        total_err = 0.0
        
        for inputs, outputs in data:
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            self.predict(inputs)
            total_err += self.get_err(outputs)
            self.backpropagate(outputs)
            
        print("Batch error: {:.3f}".format(total_err/batch_size))
        for lyr in self.layers[1:]:
            lyr.apply_batch(batch_size, rate)
            
def main(cmd, cfg_file, data_file):
    batch_size = 32
    
    with open(cfg_file, 'r') as a_file:
        cfg = json.loads(a_file.read())
    with open(data_file, 'r') as d_file:
        data = json.loads(d_file.read())
    
    num_samples = len(data)
    num_training = num_samples*3//4
    num_validation = num_samples - num_training
    
    if 'wgts' in cfg:
        wgts = [np.array(lyr) for lyr in cfg['wgts']]
    else:
        wgts = None
    nn = Network(cfg['arch'], cfg['err'], wgts)
    
    if cmd == 'verify':
        for inputs, outputs in data:
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            prd = nn.predict(inputs)
            print("{} vs {} for {:.6f}".format(prd, outputs,
                nn.get_err(outputs)))
            nn.validate_derivs(inputs, outputs)
    elif cmd == 'run':
        for start in range(0, num_training, batch_size):
            end = min(start + batch_size, num_training)
            print("Batch {}:{}".format(start, end))
            nn.run_batch(data[start:end], .01)
        
        vld_err = 0
        for inputs, outputs in data[num_training:]:
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            true_out = nn.predict(inputs)
            err = nn.get_err(outputs)
            vld_err += err
            # print(inputs, outputs, 'vs', true_out, err, vld_err)
            print(vld_err)
        
        print("Validation error: {:.3f}".format(vld_err/num_validation))
    
main(sys.argv[1], sys.argv[2], sys.argv[3])
