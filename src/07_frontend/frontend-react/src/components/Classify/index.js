import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';


import DataService from "../../services/DataService";
import styles from './styles';


const Classify = (props) => {
    const { classes } = props;

    console.log("================================== classify ======================================");


    const inputFile = useRef(null);

    // Component States
    const [image, setImage] = useState(null);
    const [salimage, setSalimage] = useState(null);
    const [state, setState] = useState({
        age: 5,
        location: "anterior torso",
        sex: "male"
    });
    const [prediction, setPrediction] = useState(
        null
    );


    // Setup Component
    useEffect(() => {


    }, []);

    // Handlers
    // salimage && window.location.reload(false);
    const handleImageUploadClick = () => {
        inputFile.current.click();
    }
    const handleOnImgChange = (event) => {
        setSalimage(null)
        setPrediction(null)
        console.log("------------------handleOnImgChange-----------------")
        console.log(event.target.files);
        setImage(URL.createObjectURL(event.target.files[0]));
        // setImage(event.target.files[0]);
        console.log("------------------handleOnImgChange 1-----------------")
        console.log(event.target.files[0]);
        console.log("------------------handleOnImgChange 2-----------------")
        console.log(event.target.files[0].type)
        console.log(inputFile.current.files[0].type)
        console.log(image);
    }
    const handleOnFormChange = (event) => {
        console.log(event.target.name);
        console.log(event.target.value);
        const value = event.target.value;
        setState({
            ...state,
            [event.target.name]: value,
        });
        console.log(state);
    }
    const handleOnFormSubmit = (event) => {
        event.preventDefault()
        setSalimage(null)
        const value = event.target.value;
        setState({
            ...state,
            [event.target.name]: value,
        });
        var formData = new FormData();
        formData.append("file", inputFile.current.files[0]);
        formData.append("age", state.age);
        formData.append("location", state.location);
        formData.append("sex", state.sex);
        console.log(formData);
        DataService.DermAIDClassificationPredict(formData)
            .then(function (response) {
                console.log(response.data);
                setPrediction(response.data);
            })
        console.log(prediction);
        DataService.DermAIDSaliencyMap(formData)
            .then(function (response) {
                console.log(response.data);
                console.log(response.data.saliency_map_url);
                setSalimage(response.data.saliency_map_url);
            })
        console.log(salimage);

    }


    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="md" className={classes.containerpred}>
                    <Typography variant="h5">DermAID Skin Cancer Classification</Typography>
                </Container>
                {
                    prediction &&
                        <Container maxWidth="md" className={classes.containerpred}>
                            <Typography variant="h6" gutterBottom>Classification Result</Typography>
                                <TableContainer component={Paper}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell className={classes.containerpred}>Prediction</TableCell>
                                                <TableCell className={classes.containerpred}>Probabilty</TableCell>
                                                <TableCell className={classes.containerpred}>Malignancy</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            <TableCell>{prediction.results[0].pred[0]}</TableCell>
                                            <TableCell>{prediction.results[0].pred[1]}</TableCell>
                                            {prediction.results[0].pred[2] === "Malignant" && <TableCell className={classes.containerpredred}>{prediction.results[0].pred[2]}</TableCell>}
                                            {prediction.results[0].pred[2] === "Benign" && <TableCell className={classes.containerpredgreen}>{prediction.results[0].pred[2]}</TableCell>}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                        </Container>
                }
                {
                    prediction &&
                        <Container maxWidth="md" className={classes.container}>
                            <Typography variant="h6" gutterBottom>Class Probability</Typography>
                            <Divider />
                                <TableContainer component={Paper}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell className={classes.containerpred}>Class</TableCell>
                                                <TableCell className={classes.containerpred}>Probability</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {prediction.results[0].pred_proba && prediction.results[0].pred_proba.map((itm, idx) =>
                                                <TableRow key={idx}>
                                                    <TableCell>{itm["name"]}</TableCell>
                                                    <TableCell>{itm["proba"]}</TableCell>
                                                </TableRow>
                                            )}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                        </Container>
                }
                <Container maxWidth="md" className={classes.container}>
                    { salimage && <Typography variant="h6" gutterBottom>Saliency Map (GradCAM)</Typography> }
                    <div className={classes.dropzone} onClick={() => handleImageUploadClick()}>
                        <input
                            type="file"
                            accept="image/*"
                            capture="camera"
                            on="true"
                            autoComplete="off"
                            tabIndex="-1"
                            className={classes.fileInput}
                            ref={inputFile}
                            onChange={(event) => handleOnImgChange(event)}
                        />
                        { ! salimage && <div><img className={classes.preview} src={image} /></div>}
                        { salimage && <div><img className={classes.preview} src={salimage} /></div>}
                        <div className={classes.help}>Click to take a picture or upload...</div>
                    </div>

                    <form onSubmit={handleOnFormSubmit} align={"center"}>
                        <label htmlFor="age">
                            Age
                            <select
                                name="age"
                                value={state.age}
                                onChange={handleOnFormChange}>
                                <option name={5}>5</option>
                                <option name={10}>10</option>
                                <option name={15}>15</option>
                                <option name={20}>20</option>
                                <option name={25}>25</option>
                                <option name={30}>30</option>
                                <option name={35}>35</option>
                                <option name={40}>40</option>
                                <option name={45}>45</option>
                                <option name={50}>50</option>
                                <option name={55}>55</option>
                                <option name={60}>60</option>
                                <option name={65}>65</option>
                                <option name={70}>70</option>
                                <option name={75}>75</option>
                                <option name={80}>80</option>
                                <option name={85}>85</option>
                            </select>
                        </label>
                        <label htmlFor="location">
                            Location
                            <select
                                name="location"
                                value={state.location}
                                onChange={handleOnFormChange}>
                                <option name={"anterior torso"}>anterior torso</option>
                                <option name={"head/neck"}>head/neck</option>
                                <option name={"lateral torso"}>lateral torso</option>
                                <option name={"lower extremity"}>lower extremity</option>
                                <option name={"oral/genital"}>oral/genital</option>
                                <option name={"palms/soles"}>palms/soles</option>
                                <option name={"posterior torso"}>posterior torso</option>
                                <option name={"upper extremity"}>upper extremity</option>
                            </select>
                        </label>
                        <label htmlFor="sex">
                            Sex
                            <select
                                name="sex"
                                value={state.sex}
                                onChange={handleOnFormChange}>
                                <option name={"male"}>male</option>
                                <option name={"female"}>female</option>
                            </select>
                        </label>
                        <br/>
                        <button type="submit">Analyze</button>
                    </form>
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(Classify);