from flow.networks import Network
import xml.etree.ElementTree as ElementTree
from lxml import etree

#
# default sumo probability value  TODO (ak): remove
DEFAULT_PROBABILITY = 0
# default sumo vehicle length value (in meters) TODO (ak): remove
DEFAULT_LENGTH = 5
# default sumo vehicle class class TODO (ak): remove
DEFAULT_VCLASS = 0


ADDITIONAL_NET_PARAMS={
    "length": 30,
    "num_lanes": 1,
    "speed_limit": 30,
}


class UniformRandomNetwork(Network):

    """
    E#R-X means Edge # strating from Right side edge to Itersection named 'X'
    E#X-R means Edge # strating from Itersection named 'X to Right side edge
    L=Left side edge
    T=Top side edge
    D=Down side edge
    """

    #Routes with Stotastics
    def specify_routes(self, net_params):
        rts = {

         "E#T-X": [
             (["E#T-X", "E#X-D"], 1/3),
             (["E#T-X", "E#X-L"], 1/3),
             (["E#T-X", "E#X-R"], 1/3),    
         ],
         "E#D-X": [
             (["E#D-X", "E#X-T"], 1/3),
             (["E#D-X", "E#X-R"], 1/3), 
             (["E#D-X", "E#X-L"], 1/3), 
         ],
         "E#L-X": [
             (["E#L-X", "E#X-R"], 1/3),  
             (["E#L-X", "E#X-D"], 1/3),  
             (["E#L-X", "E#X-T"], 1/3),    
         ],
         "E#R-X": [
             (["E#R-X", "E#X-L"], 1/3),
             (["E#R-X", "E#X-T"], 1/3),  
             (["E#R-X", "E#X-D"], 1/3), 
         ],        }

        return rts


    def _vehicle_type_custom(filename):
        """Import vehicle type data from a *.add.xml file.

        This is a utility function for outputting all the type of vehicle.

        Parameters
        ----------
        filename : str
            path to the vtypes.add.xml file to load

        Returns
        -------
        dict or None
            the key is the vehicle_type id and the value is a dict we've type
            of the vehicle, depart edges, depart Speed, departPos. If no
            filename is provided, this method returns None as well.
        """
        if filename is None:
            return None

        parser = etree.XMLParser(recover=True)
        tree = ElementTree.parse(filename, parser=parser)

        root = tree.getroot()
        veh_type = {}

        # this hack is meant to support the LuST network and Flow networks
        root = [root] if len(root.findall('vTypeDistribution')) == 0 \
            else root.findall('vTypeDistribution')

        for r in root:
            for vtype in r.findall('vType'):
                # TODO: make for everything
                veh_type[vtype.attrib['id']] = {
                    'vClass': vtype.attrib.get('vClass', DEFAULT_VCLASS),
                    'accel': vtype.attrib['accel'],
                    'decel': vtype.attrib['decel'],
                    'sigma': vtype.attrib['sigma'],
                    'length': vtype.attrib.get('length', DEFAULT_LENGTH),
                    'minGap': vtype.attrib['minGap'],
                    'maxSpeed': vtype.attrib['maxSpeed'],
                    'probability': vtype.attrib.get(
                        'probability', DEFAULT_PROBABILITY),
                    'speedDev': vtype.attrib['speedDev']
                }

        return veh_type

