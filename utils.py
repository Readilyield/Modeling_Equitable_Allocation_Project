### Utility functions, created by Ricky Huang

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


# def dispersion(N,r_vec,style=1):
#     # dispersion function for the equity objective
#     assert(len(r_vec)==N)
#     coef = 2/(N*(N-1)); out = 0
    
#     if style not in {1,2}:
#         print("wrong style, must be 1 or 2")
#         return 0
#     for i in range(N):
#         for j in range(i+1,N):
#             if style == 1:
#                 out += abs(r_vec[i]-r_vec[j])
#             elif style == 2:
#                 out += abs(r_vec[i]-r_vec[j])**2
#     return out*coef
    


def baselineModel(N,M,c,o,n,lam,p1,p0,alpha,
                  beta=100,rou=1e-3,gamma=1e-5,
                  d_style=1,name=None,pout=False):
    '''
       -- int: -- 
       N: number of locations
       M: number of resources
       
       -- np.ndarray: --
       c: logistics cost (N x 1)
       o: in-stock resource amount (N x 1)
       n: total demand (N x 1)
       lam: demand incidence rate (N x 1)
       p1: survival prob. with the resource (N x 1)
       p0: survival prob. with no resource (N x 1)
       
       -- params: --
       alpha: minimal coverage ratio
       rou: coefficient for equity penalty
       gamma: coefficient for logistics cost
    '''
    if name is None:
        name = "test"
    model = gp.Model(name)
    if d_style not in {1,2}:
        print("wrong style, must be 1 or 2")
        return 0
    
    if not pout:
        model.Params.LogToConsole = 0
    
    y = model.addMVar(N, lb=0, vtype=GRB.INTEGER, name="assigned")
    z = model.addMVar(N, lb=0, name="aux") # z = min{y, n-o}
    r_term = model.addMVar((N, N), lb=0, name="r_term")
    y_out = []
    
    Delta = p1 - p0
    D_coef = 2/(N*(N-1))

    
    for i in range(N):
        model.addConstr( z[i] <= y[i], name=f"min(y,n-o)_1_{i}" )
        model.addConstr( z[i] <= n[i]-o[i], name=f"min(y,n-o)_2_{i}" )
        model.addConstr( (o[i] + y[i])/(n[i] * lam) >= alpha, 
                        name=f"coverage minimal_{i}" )
    model.addConstr( gp.quicksum(y[i] for i in range(N)) <= M, 
                    name="resource limit" )
    
    
    for i in range(N):
        for j in range(i+1,N):
            if d_style == 1:
                model.addConstr( r_term[i,j] >= (o[i] + y[i])/(n[i] * lam)-(o[j] + y[j])/(n[j] * lam) )
                model.addConstr( r_term[i,j] >= (o[j] + y[j])/(n[j] * lam)-(o[i] + y[i])/(n[i] * lam) ) 
            elif d_style == 2:
                model.addConstr( r_term[i,j] == ((o[i] + y[i])/(n[i] * lam)-(o[j] + y[j])/(n[j] * lam))**2 )
    D = gp.quicksum(r_term[i,j] for i in range(N) for j in range(i+1,N))
    objective = beta*Delta@z - rou*D_coef*D - gamma*c@y
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
#     if model.status == GRB.INFEASIBLE:

    
    opm_val = model.getObjective().getValue()
    
    if pout:
        print("\n---Output:---\n")
        model.printAttr('x')
        print(f"max obj.value is {opm_val}.\n")
        
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)
    for name, val in zip(names, values):
#         print(f"{name} = {val}")
        if "assigned" in name:
            y_out.append(val)
    return np.array(y_out), opm_val





############################
'''Plot-related functions'''
############################

def plot_two_bars(df_sorted,col1,col2,
                  label1,label2,
                  color1=None,color2="orange",
                  ylabel="vaccine allocation"):
    # 1) Bar plots (descending) for each location
    plt.figure(figsize=(12, 4))
    x = np.arange(len(df_sorted))        # positions for states
    width = 0.4                          # bar width
    fig, ax1 = plt.subplots(figsize=(12, 4))
     # Second y-axis
    ax2 = ax1.twinx()
    if color1 is not None:
        bars1 = ax1.bar(
            x - width/2,
            df_sorted[col1],
            width,
            label=label1,color=color1)
    else:
        bars1 = ax1.bar(
            x - width/2,
            df_sorted[col1],
            width,
            label=label1)
    bars2 = ax2.bar(
        x + width/2,
        df_sorted[col2],
        width,
        label=label2,color=color2)

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_sorted["State"], rotation=90)
    # Axis labels
    ax1.set_ylabel(ylabel,fontsize=12)
    ax2.set_ylabel(label2,fontsize=12)
    ax1.set_title(f"{label1} & {label2}",fontsize=14)
    ax1.legend(loc="upper center")
    ax2.legend(loc="upper right")
    ax2.axis("off")
    plt.tight_layout()
    plt.show() 
    

def plot_vaccine_by_state(df_sorted,column="",title="",color=None):
    
    # 1) Bar plots (descending) for each location
    plt.figure(figsize=(12, 4))
    x = np.arange(len(df_sorted))        # positions for states
    width = 0.4                          # bar width
    fig, ax1 = plt.subplots(figsize=(12, 4))
     # Second y-axis
    ax2 = ax1.twinx()
    # Left axis: Population
    bars1 = ax1.bar(
        x - width/2,
        df_sorted["Population"],
        width,
        label="Population",color="orange")
    # Right axis: W*_1
    if color is not None:
        bars2 = ax2.bar(
            x + width/2,
            df_sorted[column],
            width,
            label=title,color=color)
    else:
        bars2 = ax2.bar(
            x + width/2,
            df_sorted[column],
            width,
            label=title)
    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_sorted["State"], rotation=90)
    # Axis labels
    ax1.set_ylabel("Population",fontsize=12)
    ax2.set_ylabel(f"{title} allocation",fontsize=12)
    ax1.set_title(f"Population & {title}",fontsize=14)
    ax1.legend(loc="upper center")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()  







############################
'''Data-related functions'''
############################


def try_int_convert(value):
    if pd.isna(value):
        return value  # keep NaN as-is
    s = str(value).strip()
    if s == "":
        return s      # keep empty strings empty

    # Remove thousand separators like "1,234" -> "1234"
    s_clean = s.replace(",", "")

    try:
        return int(s_clean)  # success: return as integer
    except ValueError:
        return value         # fail: leave the original text alone


def clean_data():
    #remove unnecessary columns and rows,
    #combine city and state allocations
    df = pd.read_csv("Dataset/COVID-19_Vaccine.csv", dtype=str)
    df2 = pd.read_csv("Dataset/labels.csv")


    df = df.drop(columns=df.columns[1])
    label_col = df.columns[0]
    label2 = df2.columns[0] 
    value_cols = df.columns[1:] 

    for col in value_cols:
        df[col] = df[col].apply(try_int_convert)
    print(df[value_cols].dtypes)

    df.loc[df[label_col]=="Illinois", value_cols] += (
        df.loc[df[label_col]=="Chicago", value_cols].values)
    df.loc[df[label_col]=="New York", value_cols] += (
        df.loc[df[label_col]=="New York City", value_cols].values)
    df.loc[df[label_col]=="Pennsylvania", value_cols] += (
        df.loc[df[label_col]=="Philadelphia", value_cols].values)

    bad_labels = df2.loc[df2[label2] != "Yes", label2]

    # Keep only those rows in the first csv
    df_filtered = df[~ df[label_col].isin(bad_labels)]
    df_filtered.columns = ["State","W1_1","W1_2","W2_1","W2_2",
                           "W3_1","W3_2","W4_1","W4_2","W5_1","W5_2",
                           "Total_1","Total_2"]
    df_filtered.to_csv("Dataset/cleaned_vaccine_data.csv", index=False)
    # change one or more column names
    
    print("--- cleaned data generated ---")
    

def extract_2020_population(xlsx_path):
    """
    From a Census-style file like Kansas.xlsx, extract the 2020
    Total (All Ages), Both Sexes, Population Estimate (as of July 1).
    Assumes the layout matches the Kansas example.
    """
    # Read without headers so we can search rows flexibly
    df = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    
    # 1) Find the row that contains the year labels (one of the cells is 2020)
    year_row_idx = df.isin([2020]).any(axis=1).idxmax()
    sex_row_idx = year_row_idx + 1

    year_row = df.iloc[year_row_idx]
    sex_row = df.iloc[sex_row_idx]

    # 2) Find the column where year == 2020 AND sex == "Both Sexes"
    mask = (year_row == 2020) & (sex_row == "Both Sexes")
    cols = df.columns[mask]
    if len(cols) == 0:
        raise ValueError(f"Could not find 2020 / Both Sexes column in {xlsx_path}")
    col = cols[0]

    # 3) Find the row where the first column is "Total" (all ages)
    first_col = df.iloc[:, 0]
    total_rows = first_col[first_col == "Total"]
    if total_rows.empty:
        raise ValueError(f"Could not find 'Total' row in {xlsx_path}")
    total_row_idx = total_rows.index[0]

    value = df.iloc[total_row_idx, col]
    return int(value)


def map_population(FOLDER_WITH_EXCEL, CSV_IN, CSV_OUT):
    # Build a mapping: State name -> 2020 population
    state_population = {}
    for path in glob.glob(os.path.join(FOLDER_WITH_EXCEL, "*.xlsx")):
        filename = os.path.basename(path)
        # Assume filename like "Kansas.xlsx" or "New_Hampshire.xlsx"
        state_name = os.path.splitext(filename)[0].replace("_", " ")
        
        try:
            pop_2020 = extract_2020_population(path)
            state_population[state_name] = pop_2020
            print(f"{state_name}: {pop_2020}")
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    # Load your previous CSV and add the population column
    df = pd.read_csv(CSV_IN)

    # Map populations by state name
    df["Population"] = df["State"].map(state_population)

    # (Optional) check any states that didn't get matched
    missing = df[df["Population"].isna()]["State"].unique()
    if len(missing) > 0:
        print("missing")
        print(missing)

    # Save the new CSV
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved updated CSV with population to: {CSV_OUT}")
    