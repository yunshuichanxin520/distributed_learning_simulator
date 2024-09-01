from distributed_learning_simulator.method.shapley_value.shapley_value_server import \
    ShapleyValueServer

from .algorithm import BiFedSVAlgorithm


class BiFedSVServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=BiFedSVAlgorithm(server=self))

    def server_client_bidirectional_selection(self, bifed_sv, expected_sv):
        """
        Server-client bidirectional selection mechanism (S-C) implementation in Python.

        Parameters:
        bifed_sv (dict): A dictionary where keys are participant indices and values are the BiFedSV values φ^(r)_i.
        expected_sv (dict): A dictionary where keys are participant indices and values are the expected SV values φ^(r)'_i.

        Returns:
        set: The set of selected participants for the next round N^(r+1).
        """
        # Initialize participant set N^(r+1) as an empty set
        round_N = set()
        if (self.round_number == 1):
            for key, value in bifed_sv:
                round_N.add(key)
        else:
            # Iterate over each participant i in N (all participants in the bifed_sv dictionary)

            for key,value in bifed_sv:
                # Check if φ^(r)_i >= 0 and φ^(r)_i >= φ^(r)'_i
                if bifed_sv[key] >= 0 and bifed_sv[key] >= sum(bifed_sv)/len(bifed_sv):#expected_sv[i]:
                    # Add participant i to N^(r+1)
                    round_N.add(key)
        self.selection_result[self.round_number] = round_N

        # Return the selected participants set N^(r+1)
        # return round_N
        #return super().server_client_bidirectional_selection()