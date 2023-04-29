import SingleLayerPerceptron
from SingleLayerPerceptron import SingleLayerPerceptron

def test(self):
        neutron = SingleLayerPerceptron(inputs=2)
        neutron.set_weights([10, 10, -15])  # AND Gate behavior

        print("Gate:")
        print("0 0 = {0:.10f}".format(neutron.run([0, 0])))
        print("0 1 = {0:.10f}".format(neutron.run([0, 1])))
        print("1 0 = {0:.10f}".format(neutron.run([1, 0])))
        print("1 1 = {0:.10f}".format(neutron.run([1, 1])))


        neutron = SingleLayerPerceptron(inputs=2)
        neutron.set_weights([15, 15, -10])  # OR Gate behavior

        print("Gate:")
        print("0 0 = {0:.10f}".format(neutron.run([0, 0])))
        print("0 1 = {0:.10f}".format(neutron.run([0, 1])))
        print("1 0 = {0:.10f}".format(neutron.run([1, 0])))
        print("1 1 = {0:.10f}".format(neutron.run([1, 1])))

        neutron = SingleLayerPerceptron(inputs=2)
        neutron.set_weights([-10, -10, 15])  # NAND Gate behavior

        print("Gate:")
        print("0 0 = {0:.10f}".format(neutron.run([0, 0])))
        print("0 1 = {0:.10f}".format(neutron.run([0, 1])))
        print("1 0 = {0:.10f}".format(neutron.run([1, 0])))
        print("1 1 = {0:.10f}".format(neutron.run([1, 1])))

        neutron = SingleLayerPerceptron(inputs=2)
        neutron.set_weights([-15, -15, 10])  # NOR Gate behavior

        print("Gate:")
        print("0 0 = {0:.10f}".format(neutron.run([0, 0])))
        print("0 1 = {0:.10f}".format(neutron.run([0, 1])))
        print("1 0 = {0:.10f}".format(neutron.run([1, 0])))
        print("1 1 = {0:.10f}".format(neutron.run([1, 1])))