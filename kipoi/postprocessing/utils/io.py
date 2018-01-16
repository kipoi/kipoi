
# simple class to save a bed file sequentially
from abc import abstractmethod

import numpy as np
import cyvcf2
import vcf
from tqdm import tqdm
from kipoi.postprocessing.utils.generic import prep_str, convert_record


class Bed_writer:
    ## At the moment
    def __init__(self, output_fname):
        self.output_fname = output_fname
        self.ofh = open(self.output_fname, "w")
    #
    def append_interval(self, chrom, start, end, id):
        chrom = "chr" + str(chrom).strip("chr")
        self.ofh.write("\t".join([chrom, str(int(start)-1), str(end), str(id)]) + "\n")
    #
    def close(self):
        self.ofh.close()
    #
    def __enter__(self):
        return self
    #
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Sync_predictons_writer(object):
    def __init__(self, model):
        self.info_tag_prefix = "KPVEP"
        if (model.info.name is None) or (model.info.name == ""):
            self.model_name = model.info.doc[:15] + ":" + model.info.version
        else:
            self.model_name = model.info.name + ":" + str(model.info.version)

        if (self.model_name is not None) or (self.model_name != ""):
            self.info_tag_prefix += "_%s" % prep_str(self.model_name)


    @abstractmethod
    def __call__(self, predictions, records):
        """
        Function that will be called by the predict function after every batch
        """
        pass


class Vcf_writer_cyvcf2(Sync_predictons_writer):
    """
    Synchronous writer of output VCF
    The reference cyvcf object here has to be the one from which the records are taken. INFO tags of this reference
    object will be modified in the process!
    """
    def __init__(self, model, reference_cyvcf2_obj, out_vcf_fpath, id_delim=":"):
        super(Vcf_writer_cyvcf2, self).__init__(model)
        # self.vcf_reader = cyvcf2.Reader(reference_vcf_path, "r")
        self.vcf_reader = reference_cyvcf2_obj
        self.out_vcf_fpath = out_vcf_fpath
        self.id_delim = id_delim
        self.prediction_labels = None
        self.column_labels = None

    def __call__(self, predictions, records):
        # First itertation: the output file has to be created and the headers defined
        if len(predictions) == 0:
            return None

        if self.prediction_labels is None:
            # setup the header
            self.prediction_labels = list(predictions.keys())
            for k in predictions:
                col_labels_here = predictions[k].columns.tolist()
                # Make sure that the column are consistent across different prediction methods
                if self.column_labels is None:
                    self.column_labels = col_labels_here
                else:
                    if not np.all(np.array(self.column_labels) == np.array(col_labels_here)):
                        raise Exception(
                            "Prediction columns are not identical for methods %s and %s" % (predictions.keys()[0], k))
                # Add the tag to the vcf file
                #"##INFO=<ID={ID},Number={Number},Type={Type},Description=\"{Description}\">".format(**adict)
                info_tag = {"ID":self.info_tag_prefix + "_%s" % k.upper(),
                            "Number":None, "Type":"String",
                            "Description":"%s SNV effect prediction. Prediction from model outputs: %s" % (
                                                                  k.upper(), "|".join(self.column_labels))}
                self.vcf_reader.add_info_to_header(info_tag)
            # Now we can also create the vcf writer
            self.vcf_writer = cyvcf2.Writer(self.out_vcf_fpath, self.vcf_reader)
        else:
            if (len(predictions) != len(self.prediction_labels)) or not all([k in predictions for k in self.prediction_labels]):
                raise Exception("Predictions are not consistent across batches")
            for k in predictions:
                col_labels_here = predictions[k].columns.tolist()
                if not np.all(np.array(self.column_labels) == np.array(col_labels_here)):
                    raise Exception(
                        "Prediction columns are not identical for methods %s and %s" % (self.prediction_labels[0], k))

        # sanity check that the number of records matches the prediction rows:
        for k in predictions:
            if predictions[k].shape[0] != len(records):
                raise Exception("number of records does not match number the prediction rows for prediction %s."%str(k))

        # Actually write the vcf entries.
        for pred_line, record in tqdm(enumerate(records)):
            for k in predictions:
                # In case there is a pediction for this line, annotate the vcf...
                preds = predictions[k].iloc[pred_line, :]
                info_tag = self.info_tag_prefix + "_{0}".format(k.upper())
                record.INFO[info_tag] = "|".join([str(pred) for pred in preds])
            self.vcf_writer.write_record(record)

    def close(self):
        self.vcf_writer.close()


class Vcf_writer(Sync_predictons_writer):
    def __init__(self, model, reference_vcf_path, out_vcf_fpath, id_delim=":"):
        super(Vcf_writer, self).__init__(model)
        self.vcf_reader = vcf.Reader(open(reference_vcf_path), "r")
        #self.vcf_reader = reference_vcf_obj
        self.out_vcf_fpath = out_vcf_fpath
        self.id_delim = id_delim
        self.prediction_labels = None
        self.column_labels = None

    @staticmethod
    def _generate_info_field(id, num, info_type, desc, source, version):
        return vcf.parser._Info(id, num,
                                info_type, desc,
                                source, version)

    def __call__(self, predictions, records):
        # First itertation: the output file has to be created and the headers defined
        if len(predictions) == 0:
            return None

        import pdb
        #pdb.set_trace()

        if self.prediction_labels is None:
            # setup the header
            self.prediction_labels = list(predictions.keys())
            for k in predictions:
                col_labels_here = predictions[k].columns.tolist()
                # Make sure that the column are consistent across different prediction methods
                if self.column_labels is None:
                    self.column_labels = col_labels_here
                else:
                    if not np.all(np.array(self.column_labels) == np.array(col_labels_here)):
                        raise Exception(
                            "Prediction columns are not identical for methods %s and %s" % (predictions.keys()[0], k))
                # Add the tag to the vcf file
                info_tag = self.info_tag_prefix + "_%s" % k.upper()
                self.vcf_reader.infos[info_tag] = self._generate_info_field(info_tag, None, 'String',
                                                                  "%s SNV effect prediction. Prediction from model outputs: %s" % (
                                                                  k.upper(), "|".join(self.column_labels)),
                                                                  None, None)
            # Now we can also create the vcf writer
            self.vcf_writer = vcf.Writer(open(self.out_vcf_fpath, 'w'), self.vcf_reader)
        else:
            if (len(predictions) != len(self.prediction_labels)) or (not all([k in predictions for k in self.prediction_labels])):
                raise Exception("Predictions are not consistent across batches")
            for k in predictions:
                col_labels_here = predictions[k].columns.tolist()
                if not np.all(np.array(self.column_labels) == np.array(col_labels_here)):
                    raise Exception(
                        "Prediction columns are not identical for methods %s and %s" % (self.prediction_labels[0], k))

        # sanity check that the number of records matches the prediction rows:
        for k in predictions:
            if predictions[k].shape[0] != len(records):
                raise Exception("number of records does not match number the prediction rows for prediction %s."%str(k))

        # Actually write the vcf entries.
        for pred_line, record in tqdm(enumerate(records)):
            record_vcf = convert_record(record, self.vcf_reader)
            for k in predictions:
                # In case there is a pediction for this line, annotate the vcf...
                preds = predictions[k].iloc[pred_line, :]
                info_tag = self.info_tag_prefix + "_{0}".format(k.upper())
                record_vcf.INFO[info_tag] = "|".join([str(pred) for pred in preds])
            self.vcf_writer.write_record(record_vcf)

    def close(self):
        self.vcf_writer.close()


