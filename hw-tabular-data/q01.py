import pandas as pd


def load_data():
    db = {}
    # northwind traders data as csvs
    data_dir = "./data/northwind_traders_data/"
    files = [
        "categories.csv",
        "customers.csv",
        "employee_territories.csv",
        "employees.csv",
        "order_details.csv",
        "orders.csv",
        "products.csv",
        "regions.csv",
        "shippers.csv",
        "suppliers.csv",
        "territories.csv",
    ]

    for f in files:
        table_name = f.split("/")[-1].split(".")[0]
        df = pd.read_csv(data_dir + f)
        db[table_name] = df

    return db


def compute_top_n_products(db, n=5):
    orders = db["orders"]
    products = db["products"]
    order_details = db["order_details"]

    df = order_details.merge(orders, on="orderID")
    df = df.merge(products, on="productID")
    df = df.groupby(["productID"]).sum()
    df = df.sort_values(by="quantity", ascending=False)
    df = df.merge(products, on="productID")
    df = df[["productID", "productName", "quantity"]]
    df = df.head(n)
    return df


def compute_top_n_customers(db, n=5):
    orders = db["orders"]
    customers = db["customers"]
    order_details = db["order_details"]

    df = order_details.merge(orders, on="orderID")
    df = df.merge(customers, on="customerID")
    df["total_cost"] = df["unitPrice"] * df["quantity"]
    df = df.groupby(["customerID"]).sum()
    df = df.sort_values(by="total_cost", ascending=False)
    df = df.merge(customers, on="customerID")
    df = df[["customerID", "contactName", "companyName", "total_cost"]]
    df = df.head(n)
    return df


def compute_top_n_employees(db, n=5):
    orders = db["orders"]
    employees = db["employees"]
    order_details = db["order_details"]

    df = order_details.merge(orders, on="orderID")
    df = df.merge(employees, on="employeeID")
    df["total_cost"] = df["unitPrice"] * df["quantity"]
    df = df.groupby(["employeeID"]).sum()
    df = df.sort_values(by="total_cost", ascending=False)
    df = df.merge(employees, on="employeeID")
    df = df[["employeeID", "firstName", "lastName", "total_cost"]]
    df = df.head(n)
    return df


if __name__ == "__main__":

    db = load_data()

    top_5_products = compute_top_n_products(db, n=5)
    print(top_5_products)

    top_5_customers = compute_top_n_customers(db, n=5)
    print(top_5_customers)

    top_5_employees = compute_top_n_employees(db, n=5)
    print(top_5_employees)
