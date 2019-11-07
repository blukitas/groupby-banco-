import pandas as pd


class TratamientoNulls:

    def __init__(self, dataframe):
        pass

    # Drop columns ( > 50%)
    @staticmethod
    def drop_cols(dataframe):
        dataframe.drop(columns=['lat', 'lng'])

    def fill_nan(self, dataframe):
        pass
        # Baños

    # Direcion? Drop?
    # Llenar variables:
    #   Baños, mediana para el tipo de propiedad y cantidad de habitaciones (Si cant null => entonces para tipo?
    #   Habitaciones, mediana para tipo y cantidad de baños (Si cant null => entonces para tipo?
    #   Ciudad y provincia, con el que le corresponde, si es que se puede
    #       Corregir incongruencias entre ciudades y provincias, ciudades que no se corresponden a las provincias?
    #       O es muy chico y mejor dropna?
    #   Antiguedad, tipo, ciudad, zona?
    #   Zona?
    # Cuando dropna?
