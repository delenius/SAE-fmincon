import os
from pandas import read_csv, read_excel
from ax import RangeParameter, ChoiceParameter, ParameterType, ParameterConstraint
import SAE.fmincon

def evaluate_car(ax_params):
    # Translate Ax params into native racecar params
    
    # Create a car object
    new_car = SAE.fmincon.car()
    zeros = [0.0]*39
    new_car.set_vec(zeros)
    
    # Set all the parameters
    
    # car vector with continuous and integer variables
    new_car.vector = []

    # continuous parameters with fixed bounds
    for name in ['hrw','lrw','arw','hfw','lfw','wfw','afw','hsw','lsw','wsw','asw','Prt','Pft','hc','lc','wc','tc','hia','wia']:
        value = ax_params[name]
        setattr(new_car, name, value)
        new_car.vector.append(value)

    # integer parameters
    # materials   
    for i in range(5):
        name = f'mat_{i}'
        value = ax_params[name]
        setattr(new_car, SAE.fmincon.params.at[19+i, 'variable'], SAE.fmincon.materials.at[value,'q'])
        new_car.vector.append(value)
    setattr(new_car, 'Eia', new_car.qia*1000000)  

    # rear tires
    value = ax_params['rear_tire']
    setattr(new_car, 'rear_tire', value)
    setattr(new_car, SAE.fmincon.params.at[25, 'variable'], SAE.fmincon.tires.at[new_car.rear_tire,'radius'])
    setattr(new_car, SAE.fmincon.params.at[26, 'variable'], SAE.fmincon.tires.at[new_car.rear_tire,'mass'])
    new_car.vector.append(new_car.rear_tire)

    # front tires
    value = ax_params['front_tire']
    setattr(new_car, 'front_tire', value)
    setattr(new_car, SAE.fmincon.params.at[27, 'variable'], SAE.fmincon.tires.at[new_car.front_tire,'radius'])
    setattr(new_car, SAE.fmincon.params.at[28, 'variable'], SAE.fmincon.tires.at[new_car.front_tire,'mass'])
    new_car.vector.append(new_car.front_tire)

    # engine
    value = ax_params['engine']
    setattr(new_car, 'engine', value)
    setattr(new_car, SAE.fmincon.params.at[29, 'variable'], SAE.fmincon.motors.at[new_car.engine,'Power'])
    setattr(new_car, SAE.fmincon.params.at[30, 'variable'], SAE.fmincon.motors.at[new_car.engine,'Length'])
    setattr(new_car, SAE.fmincon.params.at[31, 'variable'], SAE.fmincon.motors.at[new_car.engine,'Height'])
    setattr(new_car, SAE.fmincon.params.at[32, 'variable'], SAE.fmincon.motors.at[new_car.engine,'Torque'])
    setattr(new_car, SAE.fmincon.params.at[33, 'variable'], SAE.fmincon.motors.at[new_car.engine,'Mass'])
    new_car.vector.append(new_car.engine)

    # brakes
    value = ax_params['brakes']
    setattr(new_car, 'brakes', value)
    setattr(new_car, SAE.fmincon.params.at[34, 'variable'], SAE.fmincon.brakes.at[new_car.brakes,'rbrk'])
    setattr(new_car, SAE.fmincon.params.at[35, 'variable'], SAE.fmincon.brakes.at[new_car.brakes,'qbrk'])
    setattr(new_car, SAE.fmincon.params.at[36, 'variable'], SAE.fmincon.brakes.at[new_car.brakes,'lbrk'])
    setattr(new_car, SAE.fmincon.params.at[37, 'variable'], SAE.fmincon.brakes.at[new_car.brakes,'hbrk'])
    setattr(new_car, SAE.fmincon.params.at[38, 'variable'], SAE.fmincon.brakes.at[new_car.brakes,'wbrk'])
    setattr(new_car, SAE.fmincon.params.at[39, 'variable'], SAE.fmincon.brakes.at[new_car.brakes,'tbrk'])
    new_car.vector.append(new_car.brakes)

    # suspension
    value = ax_params['suspension']
    setattr(new_car, 'suspension', value)
    setattr(new_car, SAE.fmincon.params.at[40, 'variable'], SAE.fmincon.suspension.at[new_car.suspension,'krsp'])
    setattr(new_car, SAE.fmincon.params.at[41, 'variable'], SAE.fmincon.suspension.at[new_car.suspension,'crsp'])
    setattr(new_car, SAE.fmincon.params.at[42, 'variable'], SAE.fmincon.suspension.at[new_car.suspension,'mrsp'])
    setattr(new_car, SAE.fmincon.params.at[43, 'variable'], SAE.fmincon.suspension.at[new_car.suspension,'kfsp'])
    setattr(new_car, SAE.fmincon.params.at[44, 'variable'], SAE.fmincon.suspension.at[new_car.suspension,'cfsp'])
    setattr(new_car, SAE.fmincon.params.at[45, 'variable'], SAE.fmincon.suspension.at[new_car.suspension,'mfsp'])
    new_car.vector.append(new_car.suspension)

    # continuous parameters with variable bounds
    # Note: These can have illegal values, because Ax does not know the real bounds
    for name in ['wrw','yrw','yfw','ysw','ye','yc','lia','yia','yrsp','yfsp']:
        value = ax_params[name]
        setattr(new_car, name, value)

    for i in range(10):
        temp = getattr(new_car, SAE.fmincon.params.at[46+i, 'variable'])
        new_car.vector.append(temp)
    
    obj = new_car.objectives(SAE.fmincon.weights1)[0]
    
    # TODO: Add a large penalty if the parameter assignment is illegal
    return (obj, 0.0)
    

def create_params():
    ax_params=[]
    ax_constraints=[]

    # continuous parameters with fixed bounds
    for i in range(19):
        param_name = SAE.fmincon.params.at[i, 'variable']
        min_val = SAE.fmincon.params.at[i, 'min']
        max_val = SAE.fmincon.params.at[i, 'max']
        param = RangeParameter(name=param_name, parameter_type=ParameterType.FLOAT, lower=min_val, upper=max_val)
        ax_params.append(param)

    # discrete choice parameters
    
    # materials   
    for i in range(5):
        param_name = f'mat_{i}'
        param = ChoiceParameter(name=param_name, parameter_type=ParameterType.INT, values=list(range(13)))
        ax_params.append(param)
    
    # rear tire type
    param = ChoiceParameter(name='rear_tire', parameter_type=ParameterType.INT, values=list(range(7)))
    ax_params.append(param)

    # front tire type
    param = ChoiceParameter(name='front_tire', parameter_type=ParameterType.INT, values=list(range(7)))
    ax_params.append(param)

    # engine type
    param = ChoiceParameter(name='engine', parameter_type=ParameterType.INT, values=list(range(21)))
    ax_params.append(param)

    # brake type
    param = ChoiceParameter(name='brakes', parameter_type=ParameterType.INT, values=list(range(34)))
    ax_params.append(param)

    # brake type
    param = ChoiceParameter(name='suspension', parameter_type=ParameterType.INT, values=list(range(5)))
    ax_params.append(param)

    # continuous params bounded by other continuous params
    hrw_min = 0.025
    hfw_min = 0.025
    hsw_min = 0.025
    hc_min = 0.5
    lfw_min = 0.05
    hia_min = 0.1
    
    # Here, we use maximum ranges. Some of these are narrowed down below
    ax_params.append(RangeParameter(name='yrw', parameter_type=ParameterType.FLOAT, lower=.500 + hrw_min/2, upper=1.200 - hrw_min/2))
    ax_params.append(RangeParameter(name='yfw', parameter_type=ParameterType.FLOAT, lower=0.03 + hfw_min, upper=.250 - hfw_min/2))
    ax_params.append(RangeParameter(name='ysw', parameter_type=ParameterType.FLOAT, lower=0.03 + hsw_min/2, upper=.250 - hsw_min/2))
    ax_params.append(RangeParameter(name='yc', parameter_type=ParameterType.FLOAT, lower=0.03 + hc_min/2, upper=1.2 - hc_min/2))
    ax_params.append(RangeParameter(name='lia', parameter_type=ParameterType.FLOAT, lower=0.2, upper=.700 - lfw_min))
    ax_params.append(RangeParameter(name='yia', parameter_type=ParameterType.FLOAT, lower=0.03 + hia_min/2, upper=1.200 - hia_min/2))

    #      yrw > .500 + hrw/2
    # <=>  0.5*hrw -1.0*yrw < -0.5
    ax_constraints.append(ParameterConstraint(constraint_dict={'yrw': -1.0, 'hrw': 0.5}, bound=-0.5))

    #      yrw < 1.200 - hrw/2
    # <=>  0.5*hrw +1.0*yrw < 1.200
    ax_constraints.append(ParameterConstraint(constraint_dict={'yrw': 1.0, 'hrw': 0.5}, bound=1.200))

    #      yfw > 0.03 + hfw
    # <=>  -1.0*yfw + 1.0*hfw < -0.03
    ax_constraints.append(ParameterConstraint(constraint_dict={'yfw': -1.0, 'hfw': 1.0}, bound=-0.03))

    #      yfw < 0.25 - hfw/2
    # <=>  1.0*yfw + 0.5*hfw < 0.25
    ax_constraints.append(ParameterConstraint(constraint_dict={'yfw': 1.0, 'hfw': 0.5}, bound=0.25))

    #      ysw > 0.03 + hsw/2
    # <=>  -1.0*ysw + 0.5*hsw < -0.03
    ax_constraints.append(ParameterConstraint(constraint_dict={'ysw': -1.0, 'hsw': 0.5}, bound=-0.03))

    #      ysw < .250 - hsw/2
    # <=>  1.0*ysw + 0.5*hsw < 0.25
    ax_constraints.append(ParameterConstraint(constraint_dict={'ysw': 1.0, 'hsw': 0.5}, bound=0.25))

    #      yc > 0.03+hc/2
    # <=>  -1.0*yc + 0.5hc < -0.03
    ax_constraints.append(ParameterConstraint(constraint_dict={'yc': -1.0, 'hsw': 0.5}, bound=-0.03))

    #      yc < 1.2 - hc/2
    # <=>  1.0*yc + 0.5*hc < 1.2
    ax_constraints.append(ParameterConstraint(constraint_dict={'yc': 1.0, 'hc': 0.5}, bound=1.2))

    #      lia > 0.2
    # <=>  -1.0*lia < -0.2
    ax_constraints.append(ParameterConstraint(constraint_dict={'lia': -1.0}, bound=-0.2))

    #      lia < 0.7 - lfw
    # <=>  1.0*lia + 1.0*lfw < 0.7
    ax_constraints.append(ParameterConstraint(constraint_dict={'lia': 1.0, 'lfw': 1.0}, bound=0.7))

    #      yia > 0.03+hia/2
    # <=>  -1.0*yia + 0.5*hia < -0.03
    ax_constraints.append(ParameterConstraint(constraint_dict={'yia': -1.0, 'hia': 0.5}, bound=-0.03))

    #      yia < 1.2 - hia/2
    # <=>  1.0*yia + 0.5*hia < 1.2
    ax_constraints.append(ParameterConstraint(constraint_dict={'yia': 1.0, 'hia': 0.5}, bound=1.2))
    
    
    # Continuous params with variable bounds
    # These are not directly supported in Ax. We give the maximum bounds, and take care of the illegal
    # values elsewhere
    rt_min = SAE.fmincon.tires['radius'].min()
    rt_max = SAE.fmincon.tires['radius'].max()
    he_min = SAE.fmincon.motors['Height'].min()
    ax_params.append(RangeParameter(name='wrw', parameter_type=ParameterType.FLOAT, lower=0.3, upper=9 - 2*rt_min))
    ax_params.append(RangeParameter(name='ye', parameter_type=ParameterType.FLOAT, lower=0.03+he_min/2, upper=.500 - he_min/2))
    ax_params.append(RangeParameter(name='yrsp', parameter_type=ParameterType.FLOAT, lower=rt_min, upper=2*rt_max))
    ax_params.append(RangeParameter(name='yfsp', parameter_type=ParameterType.FLOAT, lower=rt_min, upper=2*rt_max))

    return ax_params, ax_constraints
