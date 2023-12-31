
const styles = theme => ({
    root: {
        flexGrow: 1,
        minHeight: "100vh"
    },
    grow: {
        flexGrow: 1,
    },
    main: {

    },
    container: {
        backgroundColor: "#ffffff",
        paddingTop: "30px",
        paddingBottom: "20px",
    },
    containerpred: {
        fontWeight: "bold",
        fontSize: "1.1rem",
        paddingTop: "30px",
        paddingBottom: "20px",
    },
    containerpredred: {
        color: "#ff0000",
        fontWeight: "bold",
        fontSize: "1.0rem",
        paddingTop: "30px",
        paddingBottom: "20px",
    },
    containerpredgreen: {
        color: "#008000",
        fontWeight: "bold",
        fontSize: "1.0rem",
        paddingTop: "30px",
        paddingBottom: "20px",
    },
    dropzone: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        margin: "20px",
        borderWidth: "2px",
        borderRadius: "2px",
        borderColor: "#cccccc",
        borderStyle: "dashed",
        backgroundColor: "#fafafa",
        outline: "none",
        transition: "border .24s ease-in-out",
        cursor: "pointer",
        backgroundImage: "url('https://storage.googleapis.com/ac215-dermaid/app-images/dermaid-background1.png')",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        minHeight: "400px",
    },
    fileInput: {
        display: "none",
    },
    preview: {
        width: "100%",
    },
    help: {
        color: "#302f2f"
    },
});

export default styles;