import {BASE_API_URL} from "./Common";

const axios = require('axios');

const DataService = {
    Init: function () {
        // Any application initialization logic comes here
    },
    DermAIDClassificationPredict: async function (formData) {
        console.log("-------------------DermAIDClassificationPredict---------------------")
        console.log(BASE_API_URL)
        for (var pair of formData.entries()) {
            console.log(pair[0] + ', ' + pair[1]);
        }
        //const headers={'Content-Type': formData.get("image").type}
        console.log('before post URL:'+BASE_API_URL+'/classify_post')
        return await axios.post(BASE_API_URL + "/classify_post", formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    },
    DermAIDSaliencyMap: async function (formData) {
        console.log("-------------------DermAIDSaliencyMap---------------------")
        console.log(BASE_API_URL)
        for (var pair of formData.entries()) {
            console.log(pair[0] + ', ' + pair[1]);
        }
        //const headers={'Content-Type': formData.get("image").type}
        console.log('before post URL:'+BASE_API_URL+'/gradcam_post')
        return await axios.post(BASE_API_URL + "/gradcam_post", formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    }
}

export default DataService;