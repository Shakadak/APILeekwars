#!/usr/bin/env python3

import json
import math
import random
import time
import sys
import APILeekwars as API
import numpy as np

#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """

        # early stopping functionality:
        best_accuracy=1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {} -> {:0>6.3f}".format(accuracy, n, accuracy / n * 100))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=False) # Was originaly True
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {} -> {:.3f}".format(accuracy, n_data, accuracy / n_data * 100))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(int(100 * self.feedforward(x)), int(100 * y))
                        for (x, y) in data]
        #for (x, y) in results:
        #    print("guessed: {:0>4.1f} <> {:0>4.1f} :expected".format(x/1, y/1))
        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load_nn(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    z_ = sigmoid(z)
    return z_ * (1 - z_)

def get_weights_from_registers(api, token, leek_id):
    r = api.leek.get_registers(leek_id, token)
    if r["success"]:
        registers = {}
        jw = ""
        for kv in r["registers"]:
            registers[kv["key"]] = kv["value"]
        for i in range(99):
            k = "{:0>3d}".format(i)
            jw += registers[k]
        ws = json.loads(jw)
        ws["bs"] = [np.array(w) for w in ws["bs"]]
        ws["ws"] = [np.array(w) for w in ws["ws"]]
        return ws

def set_weights_to_registers(api, token, leek_id, weights):
    ws = {}
    ws["bs"] = [w.tolist() for w in weights["bs"]]
    ws["ws"] = [w.tolist() for w in weights["ws"]]
    ws["sizes"] = weights["sizes"]
    jw = json.dumps(ws, separators = (',', ':'))
    if len(jw) > 495000:
        print("NN too big")
    for i in range(99):
        k = "{:0>3d}".format(i)
        size = 5000
        offset = i * size
        v = jw[offset : offset + size]
        r = api.leek.set_register(leek_id, k, v, token)
        print("{} {}".format(k, r))
        #print("set_register {} {} {}, success: {}-".format(leek_id, k, v, r["success"]))

def put_weights_in_farmer_registers(api, farmer_name, password, weights):
    farmer = api.farmer.login_token(farmer_name, password)
    if farmer["success"]:
        token = farmer["token"]
        farmer = farmer["farmer"]
        print(farmer_name)
        for leek in farmer["leeks"].values():
            print("\t" + leek["name"])
            set_weights_to_registers(api, token, leek["id"], weights)
        r = api.farmer.disconnect(token)

def init_weights(sizes):
    weights = {}
    weights["bs"] = [np.random.randn(y, 1) for y in sizes[1:]]
    weights["ws"] = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    weights["sizes"] = sizes
    return weights

def get_batch(api, token, fight):
    batch = []
    r = api.fight.get(fight)
    while not r["success"]:
        time.sleep(5)
        r = api.fight.get(fight)
        print(r["success"])

    winning_team = r["fight"]["winner"]
    print("winning_team")
    print(winning_team)
    while winning_team == -1:
        time.sleep(1)
        r = api.fight.get(fight)
        winning_team = r["fight"]["winner"]
        print(winning_team)

    return
    #if winning_team == 0:
    #    return batch

    duration = r["fight"]["report"]["duration"]
    #if duration > 40:
    #    return batch
    max_duration = 64
    quality = 0.5 / max_duration * (max_duration - duration + 1)

    winner  = {}
    for leek in r["fight"]["data"]["leeks"]:
        winner[leek["id"]] = leek["team"] == winning_team
    r = api.fight.get_logs(fight, token)
    if not "logs" in r:
        return batch
    for xs in r["logs"].values():
        turn = filter(lambda x: x[1] == 1, xs)
        turn = list(turn)
        if turn == [] or len(turn) != 1:
            print("passing a turn")
            continue
        id = turn[0][0]
        description = turn[0][2]
        if description == 'null':
            print("null description")
            continue
        description = np.array([json.loads(description)]).transpose()
        direction = [[0.5 + quality]] if winner[id] else [[0.5 - quality]]
        print("{} -> {}".format(duration, direction))
        batch.append((description, np.array(direction)))
    return batch

def notifications():
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()
    api = API.APILeekwars()

    farmer_name = "UndersizedPalmTree"
    #farmer_name = "PumpKing"
    #farmer_name = "PumpkinAreBetter"

    token = api.farmer.login_token(farmer_name, base_data["farmers"][farmer_name])["token"]
    notifs = api.notification.get_latest(15, token)

def solo_aggro(api, token, leek):
    r = api.garden.get_leek_opponents(leek, token)
    if r["success"]:
        random.seed()
        opponent = random.choice(r["opponents"])["id"]
        return api.garden.start_solo_fight(leek, opponent, token)
    return r

def farmer_aggro(api, token):
    r = api.garden.get_farmer_opponents(token)
    if r["success"]:
        random.seed()
        #opponent = random.choice(r["opponents"])["id"]
        opponent = r["opponents"][1]["id"]
        return api.garden.start_farmer_fight(opponent, token)
    return r

def generate_data(iteration):
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()
    api = API.APILeekwars()
    nb_fight = iteration

    #farmer_name = "UndersizedPalmTree"
    #farmer_name = "PumpKing"
    #farmer_name = "PumpkinAreBetter"
    farmer_name = "PumpOnlineEvolution"

    fights = get_fights()

    data = get_data()

    token = api.farmer.login_token(farmer_name, base_data["farmers"][farmer_name])["token"]
    #ws = get_weights() #_from_registers(api, token, base_data["register_leek"][farmer_name])
    r = api.farmer.disconnect(token)

    #nn = load_nn("nn.json") #Network(ws["sizes"], ws["bs"], ws["ws"])

    for i in range(nb_fight):
        print("fight attempt: {}".format(i + 1))
        token = api.farmer.login_token(farmer_name, base_data["farmers"][farmer_name])["token"]

        #r = api.garden.start_farmer_challenge(46679, token) # vs PumpkinAreBetter
        #r = api.garden.start_farmer_challenge(46725, token) # vs UndersizedPalmTree
        #r = api.garden.start_farmer_challenge(19370, token) # vs Loutreavin
        #r = api.garden.start_farmer_challenge(38357, token) # vs TheTintin
        #r = api.garden.start_farmer_challenge(15851, token) # vs Ebatsin
        #r = api.garden.start_farmer_challenge(51896, token) # vs PumpKing
        #r = api.garden.start_farmer_challenge(31190, token) # vs SiloBall

        #r = api.garden.start_solo_challenge(56422, 56422, token) # PumPrince vs PumPrince
        #r = api.garden.start_solo_challenge(56422, 3630, token) # PumPrince vs ariitea
        #r = api.garden.start_solo_challenge(56422, 12681, token) # PumPeasant vs Leekeed
        #r = api.garden.start_solo_challenge(56729, 56729, token) # PumPeasant vs PumPeasant
        #r = api.garden.start_solo_challenge(56307, 54557, token) # PumKing vs BlackFlag
        #r = api.garden.start_solo_challenge(56307, 3630, token) # PumKing vs ariitea
        #r = api.garden.start_solo_challenge(51109, 52753, token) # FutureCoconut vs LeafyMango
        #r = api.garden.start_solo_challenge(52753, 41781, token) # LeafyMango vs raymane
        #r = api.garden.start_solo_challenge(51328, 54510, token) # StemySquash vs twogether
        #r = api.garden.start_solo_challenge(51328, 23457, token) # StemySquash vs poironardo
        #r = api.garden.start_solo_challenge(52121, 52121, token) # WintySquash vs WintySquash
        #r = api.garden.start_solo_challenge(52121, 51328, token) # WintySquash vs StemySquash
        #r = api.garden.start_solo_challenge(51328, 50906, token) # StemySquash vs LeekySquash
        #r = api.garden.start_solo_challenge(51328, 2449, token) # StemySquash vs superPoireau
        #r = api.garden.start_solo_challenge(51328, 51328, token) # StemySquash vs StemySquash
        #r = api.garden.start_solo_challenge(50953, 54510, token) # UndersizedPalmTree vs twogether
        #r = api.garden.start_solo_challenge(50953, 44873, token) # UndersizedPalmTree vs Hakushi
        #r = api.garden.start_solo_challenge(50953, 50953, token) # UndersizedPalmTree vs UndersizedPalmTree
        #r = api.garden.start_solo_challenge(50953, 23122, token) # UndersizedPalmTree vs Poppol
        #r = api.garden.start_solo_challenge(50953, 24851, token) # UndersizedPalmTree vs Ppoto
        #r = api.garden.start_solo_challenge(50953, 26899, token) # UndersizedPalmTree vs Ppoiro
        #r = api.garden.start_solo_challenge(50953, 47183, token) # UndersizedPalmTree vs Jeez

        #r = solo_aggro(api, token, 50953) # UndersizedPalmTree
        #r = solo_aggro(api, token, 51109) # FutureCoconut
        #r = solo_aggro(api, token, 52360) # ElegantPineapple
        #r = solo_aggro(api, token, 52753) # LeafyMango

        #r = solo_aggro(api, token, 50906) # LeekySquash
        #r = solo_aggro(api, token, 51071) # LovelySquash
        #r = solo_aggro(api, token, 51328) # StemySquash
        #r = solo_aggro(api, token, 52121) # WintySquash

        #r = solo_aggro(api, token, 56307) # PumpKing
        #r = solo_aggro(api, token, 56422) # PumPrince
        # r = solo_aggro(api, token, 56729) # PumPeasant

        #r = solo_aggro(api, token, 66176) # SomeIterationsLeft
        #r = solo_aggro(api, token, 66343) # EvolutionWithoutMutation
        #r = solo_aggro(api, token, 66637) # NotSelected
        #r = solo_aggro(api, token, 67007) # AlmostMonteCarlo

        r = farmer_aggro(api, token)

        # r = api.ai.test_new('''{
        #     "type":"solo",
        #     "ais":{
        #         "176":{"id":220055,"name":"BrainDead"},
        #         "276":null,
        #         "50906":{"id":218446,"name":"TooBright"},
        #         "51071":{"id":218446,"name":"TooBright"},
        #         "51328":{"id":218446,"name":"TooBright"},
        #         "52121":{"id":218446,"name":"TooBright"},
        #         "-1":null,
        #         "-2":null,
        #         "-5":null,
        #         "-4":null,
        #         "-3":null,
        #         "-6":null
        #     },
        #     "team1":{
        #         "51328":{
        #             "id":51328,
        #             "name":"StemySquash",
        #             "color":"#BC8745",
        #             "capital":0,
        #             "level":300,
        #             "talent":1946,
        #             "skin":5,
        #             "hat":9,
        #             "real":true
        #         }
        #     },
        #     "team2":{
        #         "176":{
        #             "id":176,
        #             "farmer":46679,
        #             "name":"Gama",
        #             "skin":1,
        #             "hat":null,
        #             "level":301,
        #             "life":1600,
        #             "strength":400,
        #             "wisdom":200,
        #             "agility":0,
        #             "resistance":0,
        #             "science":200,
        #             "magic":0,
        #             "frequency":0,
        #             "tp":16,
        #             "mp":13,
        #             "chips":[15,14,8,4,22,29,30,31,32,33,35,11],
        #             "weapons":[107,43,47,44]
        #         }
        #     }
        # }''', token)
        # '''{"type":"solo","ais":{"176":{"id":220055,"name":"\t\t\t\t\t\t\t\tTest/BrainDead\t\t\t"},"276":null,"50906":{"id":218446,"name":"TooBright"},"51071":{"id":218446,"name":"TooBright"},"51328":{"id":220055,"name":"\t\t\t\t\t\t\t\tTest/BrainDead\t\t\t"},"52121":{"id":218446,"name":"TooBright"},"-1":null,"-2":null,"-5":null,"-4":null,"-3":null,"-6":null},"team1":{"51328":{"id":51328,"name":"StemySquash","color":"#EB9206","capital":0,"level":300,"talent":1946,"skin":5,"hat":9,"real":true}},"team2":{"176":{"id":176,"farmer":46679,"name":"Gama","skin":1,"hat":null,"level":301,"life":1600,"strength":400,"wisdom":200,"agility":0,"resistance":0,"science":200,"magic":0,"frequency":0,"tp":16,"mp":13,"chips":[15,14,8,4,22,29,30,31,32,33,96,11],"weapons":[45,42,109,43]}}}'''
        # '''{"type":"solo","ais":{"50906":{"id":220055,"name":"\t\t\t\t\t\t\t\tTest/BrainDead\t\t\t"},"51071":{"id":220055,"name":"\t\t\t\t\t\t\t\tTest/BrainDead\t\t\t"},"51328":{"id":220055,"name":"\t\t\t\t\t\t\t\tTest/BrainDead\t\t\t"},"52121":{"id":218446,"name":"\t\t\t\t\t\t\t\tTooBright\t\t\t"}},"team1":{"50906":{"id":50906,"name":"LeekySquash","color":"#FF00AA","capital":0,"level":300,"talent":1625,"skin":6,"hat":8,"real":true},"51071":{"id":51071,"name":"LovelySquash","color":"#FF00AA","capital":0,"level":300,"talent":1662,"skin":6,"hat":9,"real":true}},"team2":{"51328":{"id":51328,"name":"StemySquash","color":"#EB9206","capital":0,"level":300,"talent":2022,"skin":5,"hat":9,"real":true},"52121":{"id":52121,"name":"WintySquash","color":"#0077DC","capital":0,"level":300,"talent":1796,"skin":2,"hat":1,"real":true}}}'''
        # test_data = '''{"type":"solo","ais":{"50906":{"id":218446,"name":"Test/BrainDead"},"51071":{"id":220055,"name":"Test/BrainDead"},"51328":{"id":220055,"name":"Test/BrainDead"},"52121":{"id":218446,"name":"TooBright"}},"team1":{"50906":{"id":50906,"name":"WintySquash","color":"#FC0FC0","capital":0,"level":300,"talent":1471,"skin":2,"hat":1,"real":true}},"team2":{"51328":{"id":51328,"name":"StemySquash","color":"#EB9206","capital":0,"level":300,"talent":2022,"skin":5,"hat":9,"real":true}}}'''
        # test_data = '''"{"type":"solo","team1":{"52121":{"id":52121,"name":"WintySquash","color":"#0077DC","capital":0,"level":300,"talent":1471,"skin":2,"hat":1,"real":true},"-1":{"id":-1,"name":"Domingo","bot":true,"level":150,"skin":1,"hat":null,"tp":"10 to 20","mp":"3 to 8","frequency":100,"life":"100 to 3000","strength":"50 to 1500","wisdom":0,"agility":0,"resistance":0,"science":0,"magic":0,"chips":[],"weapons":[]}},"team2":{"51328":{"id":51328,"name":"StemySquash","color":"#EB9206","capital":0,"level":300,"talent":1148,"skin":5,"hat":9,"real":true},"-4":{"id":-4,"name":"Guj","bot":true,"level":150,"skin":4,"hat":null,"tp":"10 to 20","mp":"3 to 8","frequency":100,"life":"100 to 3000","strength":0,"wisdom":0,"agility":0,"resistance":"50 to 1500","science":0,"magic":0,"chips":[],"weapons":[]}}}'''
        # r = api.ai.test_new(test_data, token)
        if r["success"]:
            fight = r["fight"]
            # fights.append(fight)
            # put_fights(fights)
            print(fight)
            batch = get_batch(api, token, fight)
            # data.extend(batch)
            # put_data(data)
            #if batch != []:
            #    nn.SGD(batch, 64, 5, 0.00016, monitor_training_accuracy = True, lmbda = 0.01)
            #    nn.save("nn.json")
        else:
            print(r)
            exit()

        r = api.farmer.disconnect(token)
        #nn.save("nn.json")
        #w = {"sizes": nn.sizes, "bs": nn.biases, "ws": nn.weights}
        #put_weights(nn)
        #for farmer in base_data["farmers"]:
        #    put_weights_in_farmer_registers(api, farmer, base_data["farmers"][farmer], w)

def gather_data():
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()

    api = API.APILeekwars()

    fights = get_fights()

    data = []
    for farmer in base_data["farmers"]:
        token = api.farmer.login_token(farmer, base_data["farmers"][farmer])["token"]
        for fight in fights:
            print(fight)
            batch = get_batch(api, token, fight)
            data.extend(batch)
        r = api.farmer.disconnect(token)
    put_data(data)

def update_data():
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()

    api = API.APILeekwars()

    new_fights = get_new_fights()
    if new_fights == []:
        return
    fights = get_fights()

    data = get_data()
    for farmer in base_data["farmers"]:
        token = api.farmer.login_token(farmer, base_data["farmers"][farmer])["token"]
        for fight in new_fights:
            print(fight)
            batch = get_batch(api, token, fight)
            data.extend(batch)
        r = api.farmer.disconnect(token)
    fights.extend(new_fights)
    put_data(data)
    put_fights(fights)
    put_new_fights([])

def get_data():
    data = []
    with open("data.json") as fp:
        data = json.load(fp)
    return [(np.array(a), np.array(y)) for a, y in data]

def put_data(data):
    with open("data.json", "w") as fp:
        json.dump([(a.tolist(), y.tolist()) for a, y in data], fp, separators = (',', ':'))

def get_weights():
    ws = {}
    with open("weights.json") as fp:
        ws = json.load(fp)
    ws["bs"] = [np.array(w) for w in ws["bs"]]
    ws["ws"] = [np.array(w) for w in ws["ws"]]
    return ws

def put_weights(nn):
    with open("weights.json", "w") as fp:
        json.dump({"sizes": nn.sizes, "bs": [b_.tolist() for b_ in nn.biases], "ws": [w_.tolist() for w_ in nn.weights]}, fp, separators = (',', ':'))

def get_fights():
    fights = []
    with open("fights.json") as fp:
        fights = json.load(fp)
    return fights

def put_fights(fights):
    with open("fights.json", "w") as fp:
        json.dump(fights, fp, separators = (',', ':'))

def get_new_fights():
    fights = []
    with open("new_fights.json") as fp:
        fights = json.load(fp)
    return fights

def put_new_fights(fights):
    with open("new_fights.json", "w") as fp:
        json.dump(fights, fp, separators = (',', ':'))

def upload_weights():
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()
    api = API.APILeekwars()
    ws = get_weights()
    for farmer in base_data["farmers"]:
        put_weights_in_farmer_registers(api, farmer, base_data["farmers"][farmer], ws)

def mtx_to_ai(mtxs, mtx_name):
    ai = "global " + mtx_name + " = [];\n"
    for mtx in range(0, len(mtxs)):
        ai += "var " + mtx_name + str(mtx) + " =@ (" + mtx_name + "[" + str(mtx) + "] = []);\n"
        for row in range(0, len(mtxs[mtx])):
            ai += "var " + mtx_name + str(mtx) + "_" +  str(row) + " =@ (" + mtx_name + str(mtx) + "[" + str(row) + "] = []);\n"
            for element in mtxs[mtx][row]:
                if 'e' in json.dumps(element):
                    ai += "push(" + mtx_name + str(mtx) + "_"  + str(row) + ", jsonDecode(\"" + json.dumps(element) + "\"));\n"
                else:
                    ai += "push(" + mtx_name + str(mtx) + "_"  + str(row) + ", " + json.dumps(element) + ");\n"
    return ai

def upload_weights_to_ai():
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()
    api = API.APILeekwars()
    ws = get_weights()
    ws["bs"] = [w.tolist() for w in ws["bs"]]
    ws["ws"] = [w.tolist() for w in ws["ws"]]
    code = "global WEIGHTS = jsonDecode('{}');".format((json.dumps(ws)))
    # code = "global SIZES = " + str(ws["sizes"]) + ";\n"
    # code += mtx_to_ai(ws["bs"], "BIASES")
    # code += mtx_to_ai(ws["ws"], "WEIGHTS")
    # print(code)
    # exit()
    for farmer in base_data["farmers"]:
        token = api.farmer.login_token(farmer, base_data["farmers"][farmer])["token"]
        print(farmer)
        api.ai.save(base_data["network_ai"][farmer], code, token)
        r = api.farmer.disconnect(token)

def init_nn():
    sizes = [617, 20, 1]
    #bws = init_weights(sizes)
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()
    api = API.APILeekwars()
    nn = Network(sizes)
    nn.save("nn.json")
    put_weights(nn)

def train_from_data():
    random.seed()

    nn = load_nn("nn.json")

    data = get_data()
    #random.shuffle(data)
    #test_data = data[-len(data) // 10:]
    #data = data[:-len(data) // 10]
    print("dataset size = {}".format(len(data)))
    #print("test_data size = {}".format(len(test_data)))
    nn.SGD(data, 1, 200, 1, monitor_training_accuracy = True, lmbda = 2)

    put_weights(nn)
    nn.save("nn.json")

def stuff():
    fn = "leekwars.json"
    base_data = {}
    try:
        with open(fn) as f:
            base_data = json.load(f)
    except Exception as e:
        print(e)
        print("ARGH!")
        exit()
    api = API.APILeekwars()

    #farmer_name = "UndersizedPalmTree"
    farmer_name = "PumpKing"
    #farmer_name = "PumpkinAreBetter"

    token = api.farmer.login_token(farmer_name, base_data["farmers"][farmer_name])["token"]
    services = api.service.get_all(token)
    print(json.dumps(services, sort_keys = True, indent = 4))

test = '''{
    "type":"solo",
    "ais":{
        "176":null,
        "276":null,
        "50906":{"id":218446,"name":"TooBright"},
        "51071":{"id":218446,"name":"TooBright"},
        "51328":{"id":218446,"name":"TooBright"},
        "52121":{"id":218446,"name":"TooBright"}
    },
    "team1":{
        "51328":{
            "id":51328,
            "name":"StemySquash",
            "color":"#EB9206",
            "capital":0,
            "level":300,
            "talent":1946,
            "skin":5,
            "hat":9,
            "real":true
        }
    },
    "team2":{
        "176":{
            "id":176,
            "farmer":46679,
            "name":"Gama",
            "skin":1,
            "hat":null,
            "level":300,
            "life":1600,
            "strength":400,
            "wisdom":200,
            "agility":0,
            "resistance":0,
            "science":200,
            "magic":0,
            "frequency":0,
            "tp":16,
            "mp":13,
            "chips":[15,14,8,4,22,29,30,31,32,33,96,11],
            "weapons":[45,42,109,43]
        }
    }
}'''

if __name__ == '__main__':
    iterations = int((sys.argv + ['50'])[1])
    #init_nn()
    #gather_data()
    #update_data()
    generate_data(iterations)
    #train_from_data()
    #upload_weights()
    #upload_weights_to_ai()
    #stuff()
    #put_data([])
    pass
